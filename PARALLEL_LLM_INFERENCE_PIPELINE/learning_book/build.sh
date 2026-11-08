#!/bin/bash
# Builds learning_book/Project38_Learning_Book.pdf from chapters/*.md
cd "$(dirname "$0")"
pandoc cover.md chapters/*.md -f markdown+raw_html+smart --toc --toc-depth=1 -s \
  --metadata title="Project 38 — My Learning Book" -c style.css --embed-resources \
  -o /tmp/book.html 2>/dev/null || pandoc cover.md chapters/*.md -f markdown+raw_html -s --toc --toc-depth=1 --metadata title="Project 38 — My Learning Book" -c style.css --self-contained -o /tmp/book.html
google-chrome --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=Project38_Learning_Book.pdf file:///tmp/book.html 2>/dev/null
pdfinfo Project38_Learning_Book.pdf | grep Pages
