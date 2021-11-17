class AtlasWarp:
    '''
    A helper class for warping coronal brain section images to
    the ABA. It provides methods for performing affine transformation
    of a given atlas image to a section.
    '''

    def __init__( self, section, atlas ):
        '''Contructor for warp class, takes section and atlas image as np arrays'''
        self.section = section
        self.atlas = atlas
        