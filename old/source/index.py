import os
import re
from IPython.display import HTML
from collections import namedtuple

Img = namedtuple("Img", ["name", "t", "d", "k", "f"])


class Index:
    def __init__(self, base_folder):
        self.images = []
        self.base_folder = base_folder
        self.targets = dict()
        self.distributions = dict()
        self.kinds = dict()
        self.fun = dict()
        pattern = re.compile(r"[0-9]+(_[0-9]+)+")
        image_pattern = re.compile(r"[A-z]+.png")
        for t in os.listdir(base_folder):
            if (self.is_dir(t) and bool(pattern.match(t))):
                self.targets[t] = []
                for d in self.folder(t):
                    if self.is_dir(t, d):
                        if d not in self.distributions:
                            self.distributions[d] = []
                        for k in self.folder(t, d):
                            if self.is_dir(t, d, k):
                                if k not in self.kinds:
                                    self.kinds[k] = []
                                for i in self.folder(t, d, k):
                                    if (self.is_file(t, d, k, i) and
                                            bool(image_pattern.match(i))):
                                        self.add_img(t, d, k, i)

    def add_img(self, t, d, k, i):
        f = i.split(".png")[0]
        if f not in self.fun:
            self.fun[f] = []
        self.images.append(Img(self.rel_path(t, d, k, i), t, d, k, f))
        n = len(self.images) - 1
        self.targets[t].append(n)
        self.distributions[d].append(n)
        self.kinds[k].append(n)
        self.fun[f].append(n)

    def is_dir(self, *args):
        return (os.path.isdir(self.abs_path(*args)))

    def is_file(self, *args):
        return (os.path.isfile(self.abs_path(*args)))

    def folder(self, *args):
        return os.listdir(self.base_folder + "/" + "/".join(args))

    def abs_path(self, *args):
        return os.path.abspath(self.base_folder) + "/" + "/".join(args)

    def rel_path(self, *args):
        return self.base_folder + "/" + "/".join(args)

    def select_images(self, targets=None, distributions=None,
                      kinds=None, fun=None):
        images = list(range(len(self.images)))
        if targets is not None:
            if isinstance(targets, list):
                images = [i for t in targets for i in self.targets[t]]
            else:
                images = [i for i in self.targets[targets]]
        if distributions is not None:
            if isinstance(distributions, list):
                images = [i for d in distributions for i
                          in self.distributions[d] if i in images]
            else:
                images = [i for i in self.distributions[distributions]
                          if i in images]
        if kinds is not None:
            if isinstance(kinds, list):
                images = [i for k in kinds for i
                          in self.kinds[k] if i in images]
            else:
                images = [i for i in self.kinds[kinds]
                          if i in images]
        if fun is not None:
            if isinstance(fun, list):
                images = [i for f in fun for i
                          in self.fun[f] if i in images]
            else:
                images = [i for i in self.fun[fun]
                          if i in images]
        return [imm for i, imm in enumerate(self.images) if i in images]

    def display(self, targets=None, distributions=None, kinds=None, fun=None):
            return display_images(*self.select_images(targets, distributions,
                                                      kinds, fun))


def display_images(*images):
    return HTML("".join([("<img src=" + i.name + " />" +
                          i.name)
                         for i in images]))
