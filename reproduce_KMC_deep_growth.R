###############################################################################
##  Deep recurrent autoencoders for first-year growth phenotyping and early
##  prediction of one-year malnutrition in preterm infants (KMC cohort)
##
##  Companion R script: reproduces the full pipeline
##    (1) data preparation      (2) LCTM via lcmm
##    (3) LSTM/GRU autoencoders  (4) outcome-by-phenotype
##    (5) early prediction       (6) tables & figures
##
##
# install.packages(c("readxl","dplyr","tidyr","lcmm","mclust","aricode", "caret",
# "randomForest","gbm","pROC","ggplot2","patchwork", "caret","tibble", "data.table"))
# ##    # Deep models use Keras 3 for R (TensorFlow backend):
# #install.packages("keras3"); keras3::install_keras()
# 
# 
# install.packages("reticulate")
# library(reticulate)
# py_require("keras")
# library(keras3)
# # Install Python 3.11 (this is managed by reticulate)
# reticulate::install_python("3.11:latest")
# 
# 
# ###############################################################################


# ============================================
# MAIN SCRIPT - CORRECTED FOR MY COMPUTER
# ============================================

# Set environment (add at the very beginning)
Sys.setenv(RETICULATE_PYTHON = "D:/Anaconda3/envs/r-tensorflow/python.exe")

# Optional: Silence oneDNN warnings (add if you want cleaner output)
Sys.setenv(TF_ENABLE_ONEDNN_OPTS = 0)

# Load reticulate first
library(reticulate)

# Load all packages
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(lcmm)
  library(mclust)
  library(aricode)
  library(randomForest)
  library(gbm)
  library(pROC)
  library(ggplot2)
  library(patchwork)
  library(caret)
  library(tensorflow)
  library(keras3)
})

# Set seeds
set.seed(42) 
tensorflow::set_random_seed(42)

# Verify environment (optional - can remove after confirming)
cat("✓ Python:", py_config()$python, "\n")
cat("✓ TensorFlow:", tf$version$VERSION, "\n")
cat("✓ Keras:", keras$`__version__`, "\n")


DATA_XLSX <- "KMC_results_Prof_Arashi_growth_only.xlsx"   # <- set path
OUTDIR    <- "outputs"; FIGDIR <- "figs"
dir.create(OUTDIR, showWarnings = FALSE); dir.create(FIGDIR, showWarnings = FALSE)

INDICES <- c("WAZ","LAZ","WLZ","HCZ")
KPAPER  <- c(WAZ = 3, LAZ = 3, WLZ = 3, HCZ = 2)
GRID    <- list(WAZ = seq(32,88,2), LAZ = seq(38,88,2),
                WLZ = seq(40,88,2), HCZ = seq(38,88,2))
NAMEORDER <- list(                                  # ordered high -> low attained z
  WAZ = c("WAZ catch-up","gradual WAZ gain","WAZ faltering"),
  LAZ = c("LAZ catch-up","gradual LAZ gain","LAZ faltering"),
  WLZ = c("WLZ catch-up","WLZ maintenance","WLZ faltering"),
  HCZ = c("HCZ gain","HCZ maintenance"))

###############################################################################
## 1. DATA PREPARATION
###############################################################################
raw <- read_excel(DATA_XLSX) %>%
  mutate(infant = sub("-.*","", Visit_ID),
         visit  = as.integer(sub(".*-","", Visit_ID))) %>%
  arrange(infant, PMA_W)

cut <- 50
raw <- raw %>% mutate(
  WAZ = ifelse(PMA_W < cut & !is.na(F13_W_Z),  F13_W_Z,  CA_WAZ),
  LAZ = ifelse(PMA_W < cut & !is.na(F13_L_Z),  F13_L_Z,  CA_HAZ),
  HCZ = ifelse(PMA_W < cut & !is.na(F13_HC_Z), F13_HC_Z, CA_HCZ),
  WLZ = suppressWarnings(as.numeric(CA_WHZ)),
  BMIZ= suppressWarnings(as.numeric(CA_BAZ)))

## analysis window: PMA 200-650 days; >=3 measurements
win <- raw %>% filter(PMA_W >= 200/7, PMA_W <= 650/7) %>%
  group_by(infant) %>% filter(n() >= 3) %>% ungroup()
ids <- sort(unique(win$infant))

## baseline covariates (birth row = lowest PMA)
birth <- raw %>% group_by(infant) %>% slice_min(PMA_W, n = 1, with_ties = FALSE) %>% ungroup()
base <- tibble(infant = ids) %>%
  left_join(birth %>% transmute(infant, sex_male = as.integer(!is.na(SEX) & SEX == "Male"),
                                GA, BWZ = F13_W_Z), by = "infant")
early <- raw %>% filter(PMA_W < 50, !is.na(F13_W_Z)) %>%
  group_by(infant) %>% slice_max(PMA_W, n = 1, with_ties = FALSE) %>%
  ungroup() %>% transmute(infant, earlyW = F13_W_Z)
base <- base %>% left_join(early, by = "infant") %>%
  mutate(earlyWAZgain = earlyW - BWZ)

## one-year outcomes at last visit in window
last <- win %>% group_by(infant) %>% slice_max(PMA_W, n = 1, with_ties = FALSE) %>% ungroup()
base <- base %>% left_join(
  last %>% transmute(infant,
    underweight = as.integer(!is.na(WAZ) & WAZ < -2), stunting = as.integer(!is.na(LAZ) & LAZ < -2),
    wasting = as.integer(!is.na(WLZ) & WLZ < -2),     overweight = as.integer(!is.na(BMIZ) & BMIZ > 2),
    CA_M_last = CA_M, WAZ_last = WAZ, LAZ_last = LAZ,
    WLZ_last = WLZ, HCZ_last = HCZ, BMIZ_last = BMIZ), by = "infant") %>%
  filter(!is.na(BWZ), !is.na(GA), !is.na(earlyWAZgain))
ids <- base$infant
cat(sprintf("Infants: %d | outcomes U/S/W/O = %d/%d/%d/%d\n", nrow(base),
  sum(base$underweight), sum(base$stunting), sum(base$wasting), sum(base$overweight)))

## interpolate each infant's trajectory onto a regular grid (flat-hold outside)
grid_matrix <- function(z) {
  g <- GRID[[z]]; M <- matrix(NA_real_, length(ids), length(g))
  for (i in seq_along(ids)) {
    d <- win %>% filter(infant == ids[i], !is.na(.data[[z]])) %>% arrange(PMA_W)
    if (nrow(d) >= 2) M[i, ] <- approx(d$PMA_W, d[[z]], xout = g, rule = 2)$y
  }
  cm <- colMeans(M, na.rm = TRUE)
  for (j in seq_len(ncol(M))) M[is.na(M[, j]), j] <- cm[j]
  M
}

###############################################################################
## 2. LATENT CLASS TRAJECTORY MODELLING (reference, via lcmm)
###############################################################################
long_index <- function(z)
  win %>% filter(!is.na(.data[[z]])) %>%
  transmute(infant_num = match(infant, ids), pma = PMA_W, z = .data[[z]])

fit_lctm <- function(z, k) {
  dat <- as.data.frame(long_index(z))
  m1 <- hlme(z ~ pma + I(pma^2), random = ~1, subject = "infant_num",
             ng = 1, data = dat, verbose = FALSE)
  if (k == 1) return(m1)
  gridsearch(rep = 30, maxiter = 20, minit = m1,
    hlme(z ~ pma + I(pma^2), random = ~1, subject = "infant_num",
         ng = k, mixture = ~ pma + I(pma^2), data = dat, verbose = FALSE))
}

lctm <- list()
for (z in INDICES) {
  m <- fit_lctm(z, KPAPER[[z]]); g <- GRID[[z]]
  ## class-specific predicted trajectory on the grid -> order by attained z
  pr <- predictY(m, newdata = data.frame(pma = g), var.time = "pma")$pred
  ends <- pr[nrow(pr), ]; ord <- order(ends, decreasing = TRUE)
  cl <- m$pprob[order(m$pprob$infant_num), "class"]
  remap <- setNames(NAMEORDER[[z]], ord)
  lab <- factor(remap[as.character(cl)], levels = NAMEORDER[[z]])
  ## relative entropy
  pp <- as.matrix(m$pprob[order(m$pprob$infant_num),
                          grep("^prob", names(m$pprob))])
  ent <- 1 + sum(pp * log(pp + 1e-12)) / (nrow(pp) * log(KPAPER[[z]]))
  lctm[[z]] <- list(label = lab, grid = g, curves = pr[, ord, drop = FALSE],
                    entropy = ent, model = m)
  cat(sprintf("%s: entropy=%.2f  sizes: %s\n", z, ent,
      paste(sprintf("%s=%d", NAMEORDER[[z]], table(lab)[NAMEORDER[[z]]]), collapse=", ")))
}

###############################################################################
## 3. DEEP RECURRENT AUTOENCODERS (LSTM / GRU) FOR PHENOTYPING
###############################################################################
build_ae <- function(T, cell = c("LSTM","GRU"), H = 16, L = 4) {
  cell <- match.arg(cell)
  inp <- layer_input(shape = c(T, 1))
  enc <- if (cell == "LSTM") layer_lstm(inp, units = H) else layer_gru(inp, units = H)
  lat <- layer_dense(enc, units = L, name = "latent")
  d   <- layer_repeat_vector(lat, T)
  d   <- if (cell == "LSTM") layer_lstm(d, units = H, return_sequences = TRUE)
         else                layer_gru (d, units = H, return_sequences = TRUE)
  out <- layer_time_distributed(d, layer_dense(units = 1))
  list(ae = keras_model(inp, out), enc = keras_model(inp, lat))
}

deep <- list(LSTM = list(), GRU = list()); agree <- list()
SEEDS <- c(13, 21, 34)
for (cell in c("LSTM","GRU")) {
  for (z in INDICES) {
    M <- grid_matrix(z); Tn <- ncol(M)
    Xn <- scale(M); Xn[is.na(Xn)] <- 0
    Xarr <- array(Xn, dim = c(nrow(Xn), Tn, 1))
    aris <- c(); best <- NULL
    for (s in SEEDS) {
      tensorflow::set_random_seed(s)
      nets <- build_ae(Tn, cell)
      compile(nets$ae, optimizer = optimizer_adam(learning_rate = 0.01), loss = "mse")
      fit(nets$ae, Xarr, Xarr, epochs = 400, batch_size = nrow(Xn), verbose = 0)
      emb <- predict(nets$enc, Xarr, verbose = 0)
      mc  <- Mclust(emb, G = KPAPER[[z]], verbose = FALSE)
      clr <- mc$classification  
      ends <- sapply(sort(unique(clr)), function(c) colMeans(M[clr == c, , drop = FALSE])[Tn])
      ord  <- order(ends, decreasing = TRUE)
      remap <- setNames(NAMEORDER[[z]], sort(unique(clr))[ord])
      lab <- factor(remap[as.character(clr)], levels = NAMEORDER[[z]])
      mse <- as.numeric(evaluate(nets$ae, Xarr, Xarr, verbose = 0))
      aris <- c(aris, adjustedRandIndex(lctm[[z]]$label, lab))
      if (is.null(best) || mse < best$mse)
        best <- list(label = lab, mse = mse, grid = GRID[[z]], M = M)
    }
    deep[[cell]][[z]] <- best
    agree[[paste(cell, z)]] <- c(ari_mean = mean(aris), ari_sd = sd(aris),
      nmi = aricode::NMI(as.integer(lctm[[z]]$label), as.integer(best$label)))
    cat(sprintf("%s %s: ARI=%.2f (%.2f)  recon_mse=%.3f\n",
        cell, z, mean(aris), sd(aris), best$mse))
  }
}

###############################################################################
## 4. ONE-YEAR OUTCOME PREVALENCE BY PHENOTYPE
###############################################################################
prev_by_pheno <- function(label_list) {
  out <- list()
  for (z in INDICES) for (nm in NAMEORDER[[z]]) {
    m <- label_list[[z]] == nm
    out[[length(out)+1]] <- data.frame(Index = z, Phenotype = nm, n = sum(m),
      Underweight = round(100*mean(base$underweight[m]),1),
      Stunting    = round(100*mean(base$stunting[m]),1),
      Wasting     = round(100*mean(base$wasting[m]),1),
      Overweight  = round(100*mean(base$overweight[m]),1))
  }
  do.call(rbind, out)
}
T3 <- prev_by_pheno(lapply(lctm, `[[`, "label"))
write.csv(T3, file.path(OUTDIR, "T3_outcome_by_phenotype.csv"), row.names = FALSE)

###############################################################################
## 5. EARLY PREDICTION OF ONE-YEAR MALNUTRITION (<50 wk PMA)
###############################################################################
OUTCOMES <- c("underweight","stunting","wasting","overweight")
EG <- seq(30, 50, 2)
early_seq <- t(sapply(ids, function(inf) {
  d <- raw %>% filter(infant == inf, PMA_W < 50, !is.na(F13_W_Z)) %>% arrange(PMA_W)
  if (nrow(d) < 2) rep(base$BWZ[base$infant == inf], length(EG))
  else approx(d$PMA_W, d$F13_W_Z, xout = EG, rule = 2)$y
}))
Xtab <- as.matrix(base[, c("BWZ","earlyWAZgain","GA","sex_male")])
STAT <- as.matrix(base[, c("GA","sex_male")])

make_folds <- function(y, k = 5) createFolds(factor(y), k = k, returnTrain = FALSE)

oof_tab <- function(model, y) {
  folds <- make_folds(y); p <- numeric(length(y))
  for (te in folds) {
    tr   <- setdiff(seq_along(y), te)
    Xtr0 <- Xtab[tr, , drop = FALSE]; Xte0 <- Xtab[te, , drop = FALSE]
    ctr  <- colMeans(Xtr0); sdv <- apply(Xtr0, 2, sd); sdv[sdv == 0] <- 1
    Xtr  <- scale(Xtr0, center = ctr, scale = sdv)
    Xte  <- scale(Xte0, center = ctr, scale = sdv)
    dtr  <- as.data.frame(Xtr); dtr$.y <- y[tr]
    if (model == "logistic") {
      w   <- ifelse(y[tr] == 1, sum(y[tr] == 0) / sum(y[tr] == 1), 1)
      fit <- suppressWarnings(glm(.y ~ ., data = dtr, family = binomial, weights = w))
      p[te] <- predict(fit, as.data.frame(Xte), type = "response")
    } else if (model == "rf") {
      fit <- randomForest(x = Xtr, y = factor(y[tr], levels = c(0, 1)), ntree = 300)
      p[te] <- predict(fit, Xte, type = "prob")[, "1"]
    } else if (model == "gbm") {
      fit <- gbm(.y ~ ., data = dtr, distribution = "bernoulli",
                 n.trees = 300, interaction.depth = 2, verbose = FALSE)
      p[te] <- suppressWarnings(predict(fit, as.data.frame(Xte), n.trees = 300, type = "response"))
    }
  }
  p
}

oof_deep <- function(cell, y) {
  folds <- make_folds(y); p <- numeric(length(y)); Tn <- length(EG)
  for (te in folds) {
    tr    <- setdiff(seq_along(y), te)
    es_tr <- early_seq[tr, , drop = FALSE]; es_te <- early_seq[te, , drop = FALSE]
    st_tr <- STAT[tr, , drop = FALSE];      st_te <- STAT[te, , drop = FALSE]
    mu <- mean(es_tr); sdv <- sd(es_tr); if (sdv == 0) sdv <- 1
    sm <- colMeans(st_tr); ss <- apply(st_tr, 2, sd); ss[ss == 0] <- 1
    Xtr <- array((es_tr - mu) / sdv, dim = c(length(tr), Tn, 1))
    Xte <- array((es_te - mu) / sdv, dim = c(length(te), Tn, 1))
    Str <- sweep(sweep(st_tr, 2, sm), 2, ss, "/")
    Ste <- sweep(sweep(st_te, 2, sm), 2, ss, "/")
    seq_in <- layer_input(shape = c(Tn, 1)); st_in <- layer_input(shape = c(ncol(STAT)))
    h <- if (cell == "LSTM") layer_lstm(seq_in, units = 16) else layer_gru(seq_in, units = 16)
    z <- layer_concatenate(list(h, st_in)) %>% layer_dense(16, activation = "relu") %>%
         layer_dropout(0.2) %>% layer_dense(1, activation = "sigmoid")
    clf <- keras_model(list(seq_in, st_in), z)
    cw  <- list(`0` = 1, `1` = as.numeric(sum(y[tr] == 0) / max(sum(y[tr] == 1), 1)))
    compile(clf, optimizer = optimizer_adam(learning_rate = 0.01), loss = "binary_crossentropy")
    fit(clf, list(Xtr, Str), y[tr], epochs = 160, batch_size = length(tr),
        class_weight = cw, verbose = 0)
    p[te] <- as.numeric(predict(clf, list(Xte, Ste), verbose = 0))
  }
  p
}

pred_rows <- list()
for (o in OUTCOMES) {
  y <- base[[o]]
  preds <- list(Logistic = oof_tab("logistic", y), RandomForest = oof_tab("rf", y),
                GradBoost = oof_tab("gbm", y),
                LSTM = oof_deep("LSTM", y), GRU = oof_deep("GRU", y))
  for (mdl in names(preds)) {
    r <- pROC::roc(y, preds[[mdl]], quiet = TRUE)
    ci <- as.numeric(pROC::ci.auc(r, method = "bootstrap", boot.n = 2000))
    pred_rows[[length(pred_rows)+1]] <- data.frame(
      Outcome = o, Model = mdl,
      AUROC = sprintf("%.3f (%.3f-%.3f)", as.numeric(r$auc), ci[1], ci[3]))
  }
}
T4 <- do.call(rbind, pred_rows)
write.csv(T4, file.path(OUTDIR, "T4_prediction.csv"), row.names = FALSE)

###############################################################################
## 6. TABLES & FIGURES
###############################################################################
## Table 1 - cohort characteristics
ms <- function(x) sprintf("%.2f (%.2f)", mean(x, na.rm=TRUE), sd(x, na.rm=TRUE))
np <- function(x) sprintf("%d (%.1f)", sum(x), 100*mean(x))
T1 <- tibble::tribble(~Characteristic, ~Value,
  "n", as.character(nrow(base)),
  "Male, n (%)", np(base$sex_male),
  "Gestational age, wk", ms(base$GA),
  "Birth weight z (Fenton)", ms(base$BWZ),
  "Early WAZ change to <50 wk", ms(base$earlyWAZgain),
  "Age at 1-yr, mo", ms(base$CA_M_last),
  "1-yr WAZ", ms(base$WAZ_last), "1-yr LAZ", ms(base$LAZ_last),
  "1-yr WLZ", ms(base$WLZ_last), "1-yr HCZ", ms(base$HCZ_last),
  "Underweight, n (%)", np(base$underweight), "Stunting, n (%)", np(base$stunting),
  "Wasting, n (%)",     np(base$wasting),     "Overweight, n (%)", np(base$overweight))
write.csv(T1, file.path(OUTDIR, "T1_characteristics.csv"), row.names = FALSE)

## Table 2 - phenotype agreement
T2 <- do.call(rbind, lapply(INDICES, function(z) data.frame(
  Index = z, Classes = KPAPER[[z]], Entropy = round(lctm[[z]]$entropy, 2),
  ARI_LSTM = sprintf("%.2f (%.2f)", agree[[paste("LSTM",z)]]["ari_mean"], agree[[paste("LSTM",z)]]["ari_sd"]),
  ARI_GRU  = sprintf("%.2f (%.2f)", agree[[paste("GRU",z)]]["ari_mean"],  agree[[paste("GRU",z)]]["ari_sd"]),
  NMI_LSTM = round(agree[[paste("LSTM",z)]]["nmi"],2),
  NMI_GRU  = round(agree[[paste("GRU",z)]]["nmi"],2))))
write.csv(T2, file.path(OUTDIR, "T2_phenotype_agreement.csv"), row.names = FALSE)

## Figure 1 - phenotype mean trajectories (LCTM vs LSTM vs GRU)
pal <- c("catch-up"="#2ca02c","gain"="#2ca02c","gradual"="#1f77b4",
         "maintenance"="#7f7f7f","faltering"="#d62728")
role <- function(nm) {
  if (grepl("catch-up", nm)) "catch-up" else if (grepl("gradual", nm)) "gradual"
  else if (grepl("maintenance", nm)) "maintenance"
  else if (grepl("faltering", nm)) "faltering" else "gain"
}
panel <- function(z, labels, M, ttl) {
  g <- GRID[[z]]
  df <- do.call(rbind, lapply(NAMEORDER[[z]], function(nm) {
    m <- labels == nm; if (!any(m)) return(NULL)
    data.frame(age = g/4.345, z = colMeans(M[m, , drop=FALSE]), pheno = nm, role = role(nm))
  }))
  ggplot(df, aes(age, z, colour = role)) + geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = 3) + geom_hline(yintercept = -2, linetype = 2, colour="grey60") +
    scale_colour_manual(values = pal, guide = "none") + ylim(-3.6, 2.2) +
    labs(title = ttl, x = NULL, y = z) + theme_minimal(base_size = 9)
}
plots <- list()
for (z in INDICES) {
  Mz <- grid_matrix(z)
  plots[[length(plots)+1]] <- panel(z, lctm[[z]]$label, Mz, if (z=="WAZ") "LCTM" else "")
  plots[[length(plots)+1]] <- panel(z, deep$LSTM[[z]]$label, deep$LSTM[[z]]$M, if (z=="WAZ") "LSTM" else "")
  plots[[length(plots)+1]] <- panel(z, deep$GRU[[z]]$label,  deep$GRU[[z]]$M,  if (z=="WAZ") "GRU"  else "")
}
fig1 <- wrap_plots(plots, ncol = 3)
ggsave(file.path(FIGDIR, "fig1_phenotypes.png"), fig1, width = 11, height = 12, dpi = 150)

## Figure 2 - AUROC by outcome and model
auc_df <- do.call(rbind, lapply(seq_len(nrow(T4)), function(i) {
  v <- as.numeric(sub(" .*","", T4$AUROC[i]))
  data.frame(Outcome = T4$Outcome[i], Model = T4$Model[i], AUROC = v)
}))
auc_df$Model <- factor(auc_df$Model, levels = c("Logistic","RandomForest","GradBoost","LSTM","GRU"))
fig2 <- ggplot(auc_df, aes(Outcome, AUROC, fill = Model)) +
  geom_col(position = "dodge") + geom_hline(yintercept = 0.5, linetype = 3) +
  coord_cartesian(ylim = c(0.45, 1)) + theme_minimal(base_size = 11) +
  labs(title = "Prediction of 1-year malnutrition from early growth", y = "AUROC (5-fold OOF)")
ggsave(file.path(FIGDIR, "fig2_auroc.png"), fig2, width = 9, height = 4.8, dpi = 150)

cat("\nDone. Tables and figures written to", OUTDIR, "and", FIGDIR, "\n")
###############################################################################
## End of script
###############################################################################
