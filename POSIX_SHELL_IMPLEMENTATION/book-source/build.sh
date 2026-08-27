#!/bin/bash
# Rebuild SHELL_BOOK.pdf from book.md
cd "$(dirname "$0")"
pandoc book.md -o ../SHELL_BOOK.pdf \
  --pdf-engine=xelatex --top-level-division=chapter \
  --toc --toc-depth=2 -N \
  -H preamble.tex -B titlepage.tex \
  -V documentclass=book -V classoption=openright -V classoption=11pt \
  -V geometry:"paperwidth=7in,paperheight=9.5in,inner=0.95in,outer=0.78in,top=0.85in,bottom=0.9in" \
  -V linkcolor=accent -V toccolor=inkblue
echo "built ../SHELL_BOOK.pdf"
