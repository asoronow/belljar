# Imports
import csv
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from belljarGUI import Page, GuiController
import argparse

parser = argparse.ArgumentParser(description='Process z-stack images')
parser.add_argument('-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument('-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-r', '--regions', help="which regions to include in output", default='')
parser.add_argument('-s', '--structures', help="structures file", default='')
parser.add_argument('-g', '--graphical', help='provides prompts when true', default=True)

args = parser.parse_args()

class FileLocations(ttk.Frame, Page):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)

        self.parent = parent
        self.controller = controller

        label1 = ttk.Label(self, text ="All Objects File")
        label1.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        label2 = ttk.Label(self, text ="Output File")
        label2.grid(row = 1, column = 0, padx = 10, pady = 10)

        label3 = ttk.Label(self, text ="Safe Structure File")
        label3.grid(row = 2, column = 0, padx = 10, pady = 10)

        self.objectsEntry = ttk.Entry(self)
        self.objectsEntry.grid(row= 0, column = 1, padx = 10, pady = 10)

        self.resultEntry = ttk.Entry(self)
        self.resultEntry.grid(row= 1, column = 1,padx = 10, pady = 10)

        self.structuresEntry = ttk.Entry(self)
        self.structuresEntry.grid(row= 2, column = 1,padx = 10, pady = 10)

        button2 = ttk.Button(self, text="Browse", 
                                command= lambda :self.browseObjectFiles(self.objectsEntry))

        button2.grid(row=0, column=2, padx = 10, pady = 10)

        button3 = ttk.Button(self, text="Save As", 
                                command= lambda : self.saveFiles(self.resultEntry))

        button3.grid(row=1, column=2)

        button4 = ttk.Button(self, text="Browse", 
                                command= lambda :self.browseStructuresFiles(self.structuresEntry))

        button4.grid(row=2, column=2, padx = 10, pady = 10)

        button1 = ttk.Button(self, text ="Done",
                                command = lambda : collateCount(self.controller.objectsFile, self.controller.structuresFile, self.controller.resultsFile))

        button1.grid(row = 3, column = 2, padx = 10, pady = 20)
    
    def browseObjectFiles(self, entry):
        filename = filedialog.askopenfilename(initialdir = "Z:/",
                                            title = "Select a File",
                                            filetypes = (("CSV Files",
                                                            "*.csv"),
                                                        ("all files",
                                                            "*.*")))
        # Change label contents
        entry.delete(0, END)
        entry.insert(0,filename)
        self.controller.objectsFile = filename

        
    def browseStructuresFiles(self, entry):
        filename = filedialog.askopenfilename(initialdir = "../csv",
                                            title = "Select a File",
                                            filetypes = (("CSV Files",
                                                            "*.csv"),
                                                        ("all files",
                                                            "*.*")))
        # Change label contents
        entry.delete(0, END)
        entry.insert(0,filename)
        self.controller.structuresFile = filename

    def saveFiles(self, entry):
        filename = filedialog.asksaveasfilename(initialdir = "Z:/",
                                            defaultextension  = ".csv",
                                            title = "Select a File",
                                            filetypes = (("CSV Files",
                                                            "*.csv"),
                                                        ("all files",
                                                            "*.*")))
        # Change label contents
        entry.delete(0, END)
        entry.insert(0,filename)
        self.controller.resultsFile = filename
        
    def didAppear(self):
        super().didAppear(self)
        self.controller.update()

def collateCount(objectsFile, safeRegions, resultFile):
    # Reading in objects
    objects = {}
    with open(objectsFile) as objectFile:
        objectReader = csv.reader(objectFile, delimiter=";")
        next(objectReader) # Skip Line 1
        for row in objectReader:
            section = row[0]
            if objects.get(section) != None:
                if objects[section].get(row[6]) != None:
                    objects[section][row[6]] +=1
                else:
                    objects[section][row[6]] = 1
            else:
                objects[section] = {
                row[6]: 1
            }

    # Reading in regions
    regions = {}
    with open(safeRegions) as structureFile:
        structureReader = csv.reader(structureFile, delimiter=",")
        next(structureReader) # Skip Line 1
        for row in structureReader:
            regions[row[0]] = row[3]

    # Now count things up
    sums = {}
    total = 0
    for obj in objects.items():
        for data in obj[1].items():
            id, count = data[0], data[1]
            acronym = regions.get(id)
            total += count
            if acronym != None:
                if sums.get(acronym) != None:
                    sums[acronym] += count
                else:
                    sums[acronym] = count

    # Make the results more human friendly
    prettier = {}
    for item in sums.items():
        name, count = item[0], item[1]
        if "1" in name:
            new = name.replace("1", "")
            if prettier.get(new) != None:
                    prettier[new]["L1"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L1": count, "Total": count}
        elif "2/3" in name:
            new = name.replace("2/3", "")
            if prettier.get(new) != None:
                    prettier[new]["L2/3"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L2/3":count, "Total": count}
        elif "4" in name:
            new = name.replace("4", "")
            if prettier.get(new) != None:
                    prettier[new]["L4"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L4":count, "Total": count}
        elif "5" in name:
            new = name.replace("5", "")
            if prettier.get(new) != None:
                    prettier[new]["L5"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L5":count, "Total": count}
        elif "6a" in name:
            new = name.replace("6a", "")
            if prettier.get(new) != None:
                    prettier[new]["L6a"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L6a":count, "Total": count}
        elif "6b" in name:
            new = name.replace("6b", "")
            if prettier.get(new) != None:
                    prettier[new]["L6b"] = count
                    prettier[new]["Total"] += count
            else:
                prettier[new] = {"L6b":count, "Total": count}
        else:
            prettier[name] = count

    # Write the results
    with open(resultFile, "w", newline='') as result:
        writer = csv.writer(result)
        head = ["Area", "Total", "L1", "L2/3", "L4", "L5", "L6a", "L6b"]
        writer.writerow(head)
        for item in sorted(prettier.items()):
            area, counts = item[0], item[1]
            if isinstance(counts, dict):
                row = [
                area,
                counts.get("Total", 0),
                counts.get("L1", 0),
                counts.get("L2/3", 0),
                counts.get("L4", 0),
                counts.get("L5", 0),
                counts.get("L6a", 0),
                counts.get("L6b", 0)
            ]
            else:
                row = [area, counts]
        
            writer.writerow(row)

if __name__ == '__main__':

    if args.graphical == True:
        globals = {
                "objectsFile": "",
                "resultsFile": "",
                "structuresFile": "",
                }

        app = GuiController(pages=[
                            FileLocations, 
                            ],
                            firstPage=FileLocations,
                            globals=globals)
        app.mainloop()
    else:
        collateCount(args.input.strip(), args.structures.strip(), args.output.strip())