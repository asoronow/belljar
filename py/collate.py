# Imports
import csv
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from pathlib import Path
from belljarGUI import Page, GuiController
import argparse
import pickle

parser = argparse.ArgumentParser(description="Process z-stack images")
parser.add_argument(
    "-o", "--output", help="output directory, only use if graphical false", default=""
)
parser.add_argument(
    "-i", "--input", help="input directory, only use if graphical false", default=""
)
parser.add_argument(
    "-r", "--regions", help="which regions to include in output", default=""
)
parser.add_argument("-s", "--structures", help="structures file", default="")
parser.add_argument(
    "-g", "--graphical", help="provides prompts when true", default=True
)

args = parser.parse_args()


class FileLocations(ttk.Frame, Page):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        self.parent = parent
        self.controller = controller

        label1 = ttk.Label(self, text="All Objects File")
        label1.grid(row=0, column=0, padx=10, pady=10)

        label2 = ttk.Label(self, text="Output File")
        label2.grid(row=1, column=0, padx=10, pady=10)

        label3 = ttk.Label(self, text="Safe Structure File")
        label3.grid(row=2, column=0, padx=10, pady=10)

        self.objectsEntry = ttk.Entry(self)
        self.objectsEntry.grid(row=0, column=1, padx=10, pady=10)

        self.resultEntry = ttk.Entry(self)
        self.resultEntry.grid(row=1, column=1, padx=10, pady=10)

        self.structuresEntry = ttk.Entry(self)
        self.structuresEntry.grid(row=2, column=1, padx=10, pady=10)

        button2 = ttk.Button(
            self,
            text="Browse",
            command=lambda: self.browseObjectFiles(self.objectsEntry),
        )

        button2.grid(row=0, column=2, padx=10, pady=10)

        button3 = ttk.Button(
            self, text="Save As", command=lambda: self.saveFiles(self.resultEntry)
        )

        button3.grid(row=1, column=2)

        button4 = ttk.Button(
            self,
            text="Browse",
            command=lambda: self.browseStructuresFiles(self.structuresEntry),
        )

        button4.grid(row=2, column=2, padx=10, pady=10)

        button1 = ttk.Button(
            self,
            text="Done",
            command=lambda: collateCount(
                self.controller.objectsFile,
                self.controller.structuresFile,
                self.controller.resultsFile,
            ),
        )

        button1.grid(row=3, column=2, padx=10, pady=20)

    def browseObjectFiles(self, entry):
        filename = filedialog.askopenfilename(
            initialdir="Z:/",
            title="Select a File",
            filetypes=(("CSV Files", "*.csv"), ("all files", "*.*")),
        )
        # Change label contents
        entry.delete(0, END)
        entry.insert(0, filename)
        self.controller.objectsFile = filename

    def browseStructuresFiles(self, entry):
        filename = filedialog.askopenfilename(
            initialdir="../csv",
            title="Select a File",
            filetypes=(("CSV Files", "*.csv"), ("all files", "*.*")),
        )
        # Change label contents
        entry.delete(0, END)
        entry.insert(0, filename)
        self.controller.structuresFile = filename

    def saveFiles(self, entry):
        filename = filedialog.asksaveasfilename(
            initialdir="Z:/",
            defaultextension=".csv",
            title="Select a File",
            filetypes=(("CSV Files", "*.csv"), ("all files", "*.*")),
        )
        # Change label contents
        entry.delete(0, END)
        entry.insert(0, filename)
        self.controller.resultsFile = filename

    def didAppear(self):
        super().didAppear(self)
        self.controller.update()


def collateCount(objectsFile, safeRegions, resultFile):
    # Reading in objects
    objects = {}
    with open(objectsFile) as objectFile:
        objectReader = csv.reader(objectFile, delimiter=";")
        next(objectReader)  # Skip Line 1
        for row in objectReader:
            section = row[0]
            if objects.get(section) != None:
                if objects[section].get(row[6]) != None:
                    objects[section][row[6]] += 1
                else:
                    objects[section][row[6]] = 1
            else:
                objects[section] = {row[6]: 1}

    # Reading in regions
    regions = {}
    with open(args.structures.strip(), "rb") as f:
        regions = pickle.load(f)
    # Now count things up
    sums = {}
    total = 0
    for obj in objects.items():
        for data in obj[1].items():
            region, count = int(data[0]), data[1]
            parent = regions[region]["parent"]
            total += count
            if "layer" in regions[region]["name"].lower():
                if parent != None:
                    if sums.get(parent) != None:
                        sums[parent] += count
                    else:
                        sums[parent] = count
            else:
                if sums.get(region) != None:
                    sums[region] += count
                else:
                    sums[region] = count

    # Write the results
    outputPath = Path(args.output.strip())
    with open(outputPath / "count_results.csv", "w", newline="") as resultFile:
        print("Writing output...", flush=True)
        lines = []
        runningTotals = {}
        for r, count in sums.items():
            if runningTotals.get(r, False):
                runningTotals[r] += count
            else:
                runningTotals[r] = count

        lines.append(["Totals"])
        for r, count in runningTotals.items():
            lines.append([regions[r]["name"], regions[r]["acronym"], count])
        # Write out the rows
        resultWriter = csv.writer(resultFile)
        resultWriter.writerows(lines)

        print("Done!", flush=True)


if __name__ == "__main__":
    if args.graphical == True:
        globals = {
            "objectsFile": "",
            "resultsFile": "",
            "structuresFile": "",
        }

        app = GuiController(
            pages=[
                FileLocations,
            ],
            firstPage=FileLocations,
            globals=globals,
        )
        app.mainloop()
    else:
        print("2", flush=True)
        collateCount(args.input.strip(), args.structures.strip(), args.output.strip())
