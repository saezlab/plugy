#!/usr/bin/env Rscript

require(ggplot2)
require(readr)
require(dplyr)
require(viridis)

infile <- '06022018_mouse_kidney_1b.txt.peaks.tsv'

d <- suppressMessages(read_tsv(infile)) %>%
    filter(discard == 'False' & sample_id > 0) %>%
    mutate(drugs_runs = paste0(drugs, '_', runs)) %>%
    arrange(drugs, runs, t0) %>%
    mutate(
        drugs = factor(drugs, levels = unique(drugs), ordered = TRUE),
        runs = factor(runs, levels = unique(runs), ordered = TRUE)
    )

p <- ggplot(d, aes(y = green, x = drugs_runs)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    # scale_color_brewer(guide = FALSE, palette = 'Set1') +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(relative units)') +
    ggtitle('Caspase3 activity upon drug combination treatment in CD13+ primary cells isolated from mouse kidney') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1)
    )

ggsave(sprintf('%s.runs.pdf', infile), device = cairo_pdf, width = 18, height = 4)


p <- ggplot(d, aes(y = green, x = drugs)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    # scale_color_brewer(guide = FALSE, palette = 'Set1') +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(relative units)') +
    ggtitle('Caspase3 activity upon drug combination treatment in CD13+ primary cells isolated from mouse kidney') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1)
    )

ggsave(sprintf('%s.comb.pdf', infile), device = cairo_pdf, width = 10, height = 4)

dd <- bind_rows(
    d %>%
        mutate(drug = drug1),
    d %>%
        mutate(drug = drug2)
)

p <- ggplot(dd, aes(y = green, x = drug)) +
    geom_boxplot(outlier.size = .5, lwd = .2) +
    # scale_color_brewer(guide = FALSE, palette = 'Set1') +
    theme_linedraw() +
    xlab('Sample (drug combination)') +
    ylab('Caspase3 activity\n(relative units)') +
    ggtitle('Caspase3 activity upon drug combination treatment in CD13+ primary cells isolated from mouse kidney') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1)
    )

ggsave(sprintf('%s.single-drug.pdf', infile), device = cairo_pdf, width = 6, height = 4)

ddd <- d %>%
    group_by(drugs) %>%
    mutate(casp3 = median(green)) %>%
    summarize_all(first)

p <- ggplot(ddd, aes(fill = casp3, x = drug1, y = drug2)) +
    geom_tile() +
    # scale_color_brewer(guide = FALSE, palette = 'Set1') +
    scale_fill_viridis(guide = guide_legend(title = 'Caspase3 activity\n(relative units)')) +
    theme_linedraw() +
    xlab('Drug #1') +
    ylab('Drug #2') +
    ggtitle('Caspase3 activity upon drug combination treatment\nin CD13+ primary cells isolated from mouse kidney') +
    theme(
        text = element_text(family = 'DINPro'),
        axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8, hjust = 1),
        panel.grid = element_blank()
    )

ggsave(sprintf('%s.heatmap.pdf', infile), device = cairo_pdf, width = 7, height = 6)
