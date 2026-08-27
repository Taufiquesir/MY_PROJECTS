# My Projects

A collection of my personal and academic projects, spanning systems programming
and machine learning.

| Project | Description | Tech |
|---|---|---|
| [POSIX Shell](POSIX_SHELL_IMPLEMENTATION/) | A Unix shell built from scratch — process control, pipelines, I/O redirection, signals and job control | C++, POSIX, GNU readline |
| [SMS Spam Classifier](SMS_SPAM_CLASSIFIER/) | Web app that classifies SMS messages as spam or ham, with a trained model and live demo | Python, scikit-learn, Streamlit |
| [Crop & Weed Detection](CROP_WEED_DETECTION/) | Object-detection system distinguishing crops from weeds to enable targeted pesticide use | YOLOv3, Darknet, OpenCV |

---

## POSIX Shell Implementation

A fully functional Unix shell in C++, written without using `system()`, `popen()`
or the C++ `filesystem` library.

**Highlights**
- `ls -l` output is **byte-identical** to the system `ls`, including block totals and column alignment
- Pipelines of arbitrary length that never hang — all pipe descriptors are closed in the parent
- Async-signal-safe `SIGCHLD` handling for reaping background jobs
- Built-in commands support redirection by saving and restoring the shell's own file descriptors
- Compiles clean under `-Wall -Wextra` with **zero warnings**

**Built-ins:** `cd`, `echo`, `pwd`, `ls`, `pinfo`, `search`, `history`, `jobs`

```bash
cd POSIX_SHELL_IMPLEMENTATION
make
./myshell
```

Documentation: [`SHELL_BOOK.pdf`](POSIX_SHELL_IMPLEMENTATION/SHELL_BOOK.pdf) is a
52-page implementation study covering every module and system call.
[`TESTING_GUIDE.md`](POSIX_SHELL_IMPLEMENTATION/TESTING_GUIDE.md) walks through
every test case with explanations.

---

## SMS Spam Classifier

A machine-learning web app that classifies SMS messages as spam or legitimate.
Text is cleaned and vectorised with TF-IDF, then classified by a model trained on
a labelled SMS dataset.

```bash
cd SMS_SPAM_CLASSIFIER
pip install -r requirements.txt
streamlit run app.py
```

---

## Crop & Weed Detection

An intelligent weed-detection system for agriculture. Weeds compete with crops
for resources, and blanket pesticide spraying leaves residue and harms the
environment — this model enables targeted application instead.

Built on **YOLOv3** with transfer learning from pretrained convolutional weights
in the Darknet framework, trained on a Kaggle dataset of 1,300 labelled sesame
crop and weed images with YOLO-format bounding boxes.

---

## Repository layout

```
MY_PROJECTS
├── POSIX_SHELL_IMPLEMENTATION/
├── SMS_SPAM_CLASSIFIER/
├── CROP_WEED_DETECTION/
└── README.md
```
