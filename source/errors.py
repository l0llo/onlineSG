class OSG_Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NonHomogeneousTuplesError(OSG_Error):
    """
    This exception is raised when in the values list passed to the game the
    tuple are correctly parsed but they have different numbers of elements
    """
    pass


class UnparsableGameError(OSG_Error):
    """
    Raised when for some reason there is no possibility to parse the game
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "It was not possible to parse " + repr(self.value)


class TuplesWrongLenghtError(OSG_Error):
    """
    Raised when the values lenght does not match the players number
    """
    pass


class UnparsablePlayerError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "It was not possible to parse " + repr(self.value)


class RowError(OSG_Error):
    def __str__(self):
        return str(self.__cause__)


class UnknownHeaderError(OSError):
    pass
