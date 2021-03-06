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


class UnknownHeaderError(OSG_Error):
    pass


class FolderExistsError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "You have selected an existent folder name: " + repr(self.value)


class NotAProbabilityError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + " is not a valid probability distribution"


class FinishedGameError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + " has already been run"


class UnparsableProfile(OSG_Error):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return "Can't parse an adversary profile: " + repr(self.e)


class NegativeProbabilityError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "There is a negative probability: " + repr(self.value)


class AlreadyFinalizedError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Player " + repr(self.e) + " has already been finalized"


class NotFinalizedError(OSG_Error):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Player " + repr(self.e) + " has't been finalized yet"


class InvArgError(OSG_Error):
    pass
