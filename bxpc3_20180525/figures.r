#!/usr/bin/env Rscript

require(ggplot2)
require(readr)
require(dplyr)
require(viridis)

infile <- 'denes_martine_1.txt.peaks.tsv'

d <- suppressMessages(read_tsv(infile)) %>%
    filter(
        discard == 'False' &
        barcode == 'False' &
        !is.na(drugs)
    ) %>%
    mutate(drugs_runs = paste0(drugs, ' #', cycle)) %>%
    arrange(drugs, cycle, t0) %>%
    mutate(
        drugs = factor(drugs, levels = unique(drugs), ordered = TRUE),
        runs = factor(cycle, levels = unique(cycle), ordered = TRUE)
    ) %>%
    group_by(runs) %>%
    mutate(
        zscore = (green - mean(green)) / sd(green)
    ) %>%
    ungroup()

p <- ggplot(d, aes(y = zscore, x = drugs)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(z-score)') +
    ggtitle('Caspase3 activity upon drug combination treatment in BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid.major = element_line(color = '#CCCCCC'),
        panel.grid.minor = element_line(color = '#CCCCCC')
    )

ggsave(sprintf('%s.drugs.runs.pdf', infile), device = cairo_pdf, width = 18, height = 4)


p <- ggplot(d, aes(y = zscore, x = drugs)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    facet_grid(. ~ runs) +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(z-score)') +
    ggtitle('Caspase3 activity upon drug combination treatment in BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid.major = element_line(color = '#CCCCCC'),
        panel.grid.minor = element_line(color = '#CCCCCC')
    )

ggsave(sprintf('%s.drugs_by-cycle.pdf', infile), device = cairo_pdf, width = 18, height = 4)


p <- ggplot(d, aes(y = zscore, x = drugs)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(z-score)') +
    ggtitle('Caspase3 activity upon\ndrug combination treatment in BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid.major = element_line(color = '#CCCCCC'),
        panel.grid.minor = element_line(color = '#CCCCCC')
    )

ggsave(sprintf('%s.comb.pdf', infile), device = cairo_pdf, width = 5, height = 4)

dd <- bind_rows(
    d %>%
        mutate(drug = drug1),
    d %>%
        mutate(drug = drug2)
)

p <- ggplot(dd, aes(y = zscore, x = drug)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    theme_linedraw() +
    xlab('Sample (single drug)') +
    ylab('Caspase3 activity\n(z-score)') +
    ggtitle('Caspase3 activity upon drug combination treatment in BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1)
    )

ggsave(sprintf('%s.single-drug.pdf', infile), device = cairo_pdf, width = 6, height = 4)

ddd <- d %>%
    group_by(drugs) %>%
    mutate(casp3 = median(zscore)) %>%
    summarize_all(first)

p <- ggplot(ddd, aes(fill = casp3, x = drug1, y = drug2)) +
    geom_tile() +
    scale_fill_viridis(guide = guide_legend(title = 'Caspase3 activity\n(z-score)')) +
    theme_linedraw() +
    xlab('Drug #1') +
    ylab('Drug #2') +
    ggtitle('Caspase3 activity upon drug combination treatment\nin BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid = element_blank()
    )

ggsave(sprintf('%s.heatmap.pdf', infile), device = cairo_pdf, width = 7, height = 6)


ddd <- d %>%
    group_by(drugs, runs) %>%
    mutate(casp3 = median(zscore)) %>%
    summarize_all(first)

p <- ggplot(ddd, aes(fill = casp3, x = drug1, y = drug2)) +
    geom_tile() +
    facet_grid(. ~ runs) +
    scale_fill_viridis(guide = guide_legend(title = 'Caspase3 activity\n(z-score)')) +
    theme_linedraw() +
    xlab('Drug #1') +
    ylab('Drug #2') +
    ggtitle('Caspase3 activity upon drug combination treatment in BxPC3 cells') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid = element_blank()
    )

ggsave(sprintf('%s.heatmap_by-cycle.pdf', infile), device = cairo_pdf, width = 10, height = 2.3)
