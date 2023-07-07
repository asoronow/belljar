# Bell Jar

![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge) ![Electron.js](https://img.shields.io/badge/Electron-191970?style=for-the-badge&logo=Electron&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white) ![Mac OS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

# Introduction

Bell Jar is a tool for neurohistology analysis of the mouse brain. It is still under active development.

# Compatability

Bell Jar's goal is be built on any platform, as such we provide release binaries for all platforms we can test. If you do not see your OS then please build from source using the provided instructions.

# Usage

See the belljar_guide.pdf included in the repository for a detailed set of instructions on how to run the program with our sample dataset. It also includes a guide to each individual tool.

# Install

To build and run from source please clone the main branch onto your local machine and run the following. Note that complete install requires internet and about 55GB of storage.

To install from a release binary:

Get a percompiled version from releases, download the most recent version for your OS.
To run simply extract the downloaded release and run the belljar executable.

Tip: Releases are found under tags on github.

To install from source:

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
