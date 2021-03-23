# Nonlinear Signal Processing
This repository contains the teaching materials for the Nonlinear Signal Processing problem classes (442.022) held during the summer term 2021 at Graz University of Technology. You can enroll via the university's online interface [here](https://online.tugraz.at/). See the [course homepage](https://www.spsc.tugraz.at/courses/nonlinear-signal-processing.html) for additional information about the corresponding lecture. The content of this repository will be continuously updated as the course progresses.

## Course Overview
This course serves as a supplement to the Nonlinear Signal Processing lecture (442.021), as we will elaborate on some of the theoretical concepts introduced in the lecture and discuss more recent applications (e.g. neural network architectures for system identification, normalizing flows, WaveNet, ...) of these principles. Finally, we will apply the gained knowledge by implementing some experiments. We will work ourselves through the content by roughly following the structure of the lecture. See the [Logistics](#logistics) for the planned schedule.

### Prerequisites
We expect you to know about undergraduate math, specifically basic probability theory and statistics (e.g. Bayes' Law, expectations, estimators, ...) and some linear algebra. Moreover, an introductory course in either machine learning or (linear) signal processing is strongly recommended. Knowledge about neural networks would be beneficial but is not necessary.  For the implementation tasks some experience with Python and the Numpy, Scipy and PyTorch packages would come in handy, but that is nothing you cannot learn during the course. Apart from that, we will try to keep the course as self-contained as possible.

### Grading Modalities
For each of the  three main topic blocks you will have to complete one homework assignment. We will publish the assignments in the [Teachcenter](https://tc.tugraz.at/main/course/view.php?id=1487). The assignments will be a mix of pen&paper exercises and implementation tasks. For each task you can earn points such that the achievable total over all three assignments will be at least 100 points (there will be additional tasks where you can earn bonus points). **We strongly encourage you to work in pairs,** as it is always helpful to have someone to discuss your problems with and allows you to split the implementation work. Both members of a group will get the same amount of total points achieved. You will get your grades according following grading scheme:

| Achieved Points P | Grade                 |
|:------------------------:|:----------------------:|
|  0 <= P <   50      | (5) Failed            |
| 50 <= P <   62     | (4) Sufficient      |
| 62 <= P <   75     | (3) Satisfactory  |
| 75 <= P <   88     | (2) Good             |
| 88 <= P <= 100  | (1) Excellent      |

A word on plagiarism: don't do it, there is no benefit in it for anyone (especially not for you). In addition to our corrections your submissions will undergo an automated plagiarism test. Should we discover any form of plagiarism (e.g. copying solution from the internet without citing its source, copying solutions from other groups, etc.) we will expel the culprits from the course.

## Logistics

We will try to cover the materials in approximately eight sessions (90 minutes each), but we can have additional sessions if we need more time. Participation is of course recommended but not a formal requirement. See the below table for the planned schedule.

| Date        | Session Content  |
|:-----------:|:-----------------|
| 19 Mar 2021 | MLS<sup>*</sup>: Introduction, System Identification |
| 26 Mar 2021 | MLS<sup>*</sup>: Cumulants: Analysis and Estimation |
| 16 Apr 2021 | MLS<sup>*</sup>: Propagation of Uncertainty, Normalizing Flows |
| 30 Apr 2021 | FMS<sup>*</sup>: Higher-order Statistics in Practice |
| 07 May 2021 | FMS<sup>*</sup>: System identification with Convolutional Neural Networks |
| 28 May 2021 | NFMS<sup>*</sup>: Fixed points and Stability Analysis |
| 04 Jun 2021 | NFMS<sup>*</sup>: Analysis of Chaotic Systems |
| 11 Jun 2021 | NFMS<sup>*</sup>: Synthesis with Recurrent Architectures, Wavenet |

<sup>* MLS = memoryless systems, FMS = fading memory systems, NFMS = non-fading memory systems</sup>

Due to the current circumstances around the Corona virus, we have to be more flexible this semester.  We will start by having our sessions virtually via the BigBlueButton tool that is available via the [TeachCenter](https://tc.tugraz.at/main/course/view.php?id=1487). Should the regulations allow for it, we will meet in person to work on our topics. In that case, registered participants will be notified via email at least one week in advance.

Regarding communication I intend to mainly communicate via a dedicated Discord server (the invite link is accessible via the TeachCenter). You can also ask your questions in the TeachCenter forum but discussions are expected to be more lively on Discord. Important announcements will be made via email to registered participants. For personal issues (e.g. regarding organization, grading, ...) you can of course contact me personally ([contact information](https://online.tugraz.at/tug_online/visitenkarte.show_vcard?pPersonenId=714AAFF2617048A1&pPersonenGruppe=3)).

## Getting Started
Follow the instructions below to get your (Linux) system ready for working on the assignments. ***We strongly recommend you to work on a Linux-based system.*** It will make your life easier. Alternatively, working on a Mac should do as well. It might be painful to get your setup right on Windows (although it's doable).

### 1. Get a local copy of this repository
To get a local copy of this repository, simply run
```
git clone https://github.com/chritoth/nonlinear-signal-processing-2021.git
```
If you are unfamiliar with Git, [Atlassian](https://www.atlassian.com/git/tutorials) provides you with some nice and compact tutorials. You could start by reading about [what Git actually is](https://www.atlassian.com/git/tutorials/what-is-git) or get a nice [cheat sheet](file:///tmp/mozilla_chri0/SWTM-2088_Atlassian-Git-Cheatsheet.pdf) for the most common Git commands.

#### Optional: Work in your own Git repository
In case you want to work in your own git repository (e.g. on Github), you can add this repository as an upstream
```
git remote add upstream https://github.com/chritoth/nonlinear-signal-processing-2021.git
```
and grab the latest updates from the course repository e.g. via
```
git pull upstream master
```

### 2. Set up your Python environment
We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to setup your Python environment which is available for Windows, Linux and MacOSX. You can get the latest version from [here](https://docs.conda.io/en/latest/miniconda.html)  or install it like so (Linux):
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Once you have Miniconda installed, create a virtual environment from the included environment description in `environment.yml` like so:
```
conda env create -f /path/to/local/nonlinear-signal-processing-2021/environment.yaml
```
Finally, activate the conda environment via 
```
conda activate nsp2021
```
and set your python path to the repository root
```
export PYTHONPATH="/path/to/local/nonlinear-signal-processing-2021/"
```

In case the environment specification in the `environment.yaml` changes after you have created you conda environment, you can simply update it via
```
conda env update -n nsp2021 -f environment.yaml
```

#### Optional: Latex for Matplotlib
If you want to use Latex in combination with Matplotlib, make sure that you have Latex distribution installed and the executables are included in your PATH. See [here](https://matplotlib.org/stable/tutorials/text/usetex.html) for more details.


### 3. Get familiar with the repository structure
Below is a schematic file tree of this repository.  You should stick to this structure when adding your work.
```
    ├── README.md           <- This readme file.
    │
    ├── environment.yml     <- The file for reproducing the Python environment necessary for running the code in this project.
    │
    ├── data                <- Example data.
    │
    ├── notebooks           <- Jupyter notebooks for running experiments and analysis.
    |
    ├── src                 <- Python source code.
    │   ├── __init__.py     <- Makes src a Python module
    │   ├── data            <- Data cleaning, I/O, ...
    │   ├── scripts         <- Scripts for running experiments.
    │   ├── utils           <- Utils for plotting, ...
    |
```


## Submission Guidelines

### Mandatory
You have to submit your assignment solutions via the [Teachcenter](https://tc.tugraz.at/main/course/view.php?id=1487). Each submission **must** include a written report in `.pdf` format, containing your results and summarizing your findings. Make sure to include a title page with the names and registration numbers of all group members. Additionally, you **must** provide your *well structured and commented* source code packed into a `.zip` file.
### Optional
Optionally, I would encourage you to use a personal GitHub account for your work and to make it publicly available **after the final submission deadline** (to avoid the temptation for plagiarism). Why? Whatever interesting job you might apply for in the future, showing that you can work with Git and produce readable and reusable code will be at the least a huge bonus. If you decide to so, please include a short description of the repository content (i.e. your work) and a link to *this* repository in a README.md file. There are tons of material about best practices for Git, readmes, etc. available online - have a look!

You could also decide to share only a subset of the assignments that you feel very confident about. In that case forget about the tag but do link back to this course repository.

## Acknowledgements
Many thanks to Christian Knoll and Gernot Kubin for helpful discussions and remarks on the conception of this course. Thanks to Daniel Uhl for pointing out issues with the submission modalities in the first version of this Readme and for suggesting to use Discord as a communication platform.

## Credits
