from tkinter import *
from tkinter import ttk
from tkinter import filedialog


class GuiController(Tk):
    '''
    A GuiController manages the paging and display of a set
    of subframes. It can hold any number of pages and allows
    navigation between them. Each time a page is presented the
    controller will call the pages didAppear method. All pages
    should inherit from the Page class. A controller may be
    configured with any args associated with Tk root views.

    Author: Alec Soronow
    Credit: Soumi Bardhan
    Date: September 21, 2021    

    Attributes: pages; a list of pages
                firstPage; initialy displayed page
                globals; a dict of global attributes to share between pages
                *args; Tk args passthrough
                **kwargs; Tk keyword args passthrough

    Methods: showPage(); displays the selected page and resizes view
    '''

    def __init__(self, pages, firstPage, globals, *args, **kwargs):
        '''Contructor to setup Tk, paging, and globals attributes'''
        Tk.__init__(self, *args, **kwargs)

        # Let pages decide how large the view should be
        self.resizable(False, False)
        self.title("Counting Application")  # Set title here
        self.args = args
        self.kwargs = kwargs
        self.pages = pages

        for key, value in globals.items():
            setattr(self, key, value)

        # creating a container
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing pages to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in self.pages:
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.showPage(firstPage)

    def showPage(self, cont):
        '''First removes all page grid layouts to allow resizing, rebuilds displayed layout'''
        for frame in self.frames.values():
            frame.grid_remove()

        frame = self.frames[cont]  # Select the desired page
        frame.tkraise()  # Make this the presented frame
        frame.grid()  # Layout its grid objects
        frame.didAppear()  # Let this object know its being displayed


class Page():
    '''
    Parent class for pages to be used with the GuiController.
    Its purpose is to clearly define page objects and provide
    default methods for more responsive applications. It presently
    generates a unique id for a page and provides a default
    didAppear method with a debug feature.

    Author: Alec Soronow
    Date: September 21, 2021

    Attributes: None

    Methods: didAppear(); a notification method, page specific actions upon display
    '''

    def didAppear(self, debug=False):
        if debug:
            print(f"Page{self.__class__} appeared")
