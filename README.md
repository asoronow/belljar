# Bell Jar

![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge) ![Electron.js](https://img.shields.io/badge/Electron-191970?style=for-the-badge&logo=Electron&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) ![Mac OS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

# Introduction

Bell Jar is a tool for neurohistology analysis of the mouse brain. It is still under active development.

# Compatability

Bell Jar's goal is be built on any platform, as such we provide release binaries for all platforms we can test. If you do not see your OS then please build from source using the provided instructions.

# Usage

See the belljar_guide.pdf included in the repository for a detailed set of instructions on how to run the program with our sample dataset. It also includes a guide to each individual tool.

# Requirements

- At least 20GB of disk space
- 32GB of memory MINIMUM (64GB recommneded)
- Intel i5 / Apple Silicon / AMD Ryzen 4th gen
- (REQUIRED) GPU with at least 6 GB of VRAM

# Install from Release

Get a percompiled version from releases, download the most recent version for your OS.
To run simply extract the downloaded release and run the belljar executable.

Note: On some OSX systems you'll need to authroize the Bell Jar process to run since code signing is not implemented.
See Apple's guide on running unsigned code, https://support.apple.com/en-us/HT202491.

Tip: Releases are also found under tags on github.

# Install from Source

To build and run from source please clone the main branch onto your local machine and run the following. Note that complete install requires internet and is about 20gb on disk.

```
// 1. Clone the main branch of the repository
git clone https://github.com/asoronow/belljar.git

// 2. If you do not have yarn, other wise skip to step 3
// Install node.js and npm if you do not have them (https://nodejs.org/en/download) then run
npm install -g yarn

// 3. Install all dependencies
// navigate to the cloned directory in your terminal and run
yarn install

// 4. Run the Electron app to use Bell Jar
yarn start
```

# How to work with annotations

If you want to use the annotations in your own workflows you can load them using python's built in pickle library and numpy. Annotations are just 32 bit usigned integer arrays, each index representing an Allen Atlas region id at that pixel in your aligned tissue. Each id is mapped to its region in the 'structure_graph.json' file availble in the csv folder of this repo.

_Note: Bell Jar uses the 'id' field not the 'atlas_id' field for its annotations._

```
import pickle
import numpy as np

with open("Annotation_MyBrain_s001.pkl", "rb") as file:
    annotation = pickle.load(file)

# Do stuff with the annotation
```
