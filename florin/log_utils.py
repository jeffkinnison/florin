import logging
import sys


class FlorinLogger(object):
    def __init__(self, quiet=False):
        self.logger = logging.getLogger('florin')

        if not quiet:
            self.handler = logging.StreamHandler(sys.stdout)
            self.formatter = logging.Formatter(
                '%(asctime)s %(name)s : %(message)s')
            self.handler.setFormatter(self.formatter)
        else:
            self.handler = logging.NullHandler()

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        self.quiet = quiet

    def __setattr__(self, key, val):
        if key == 'quiet':
            self.logger.removeHandler(self.handler)
            self.handler = logging.StreamHandler(sys.stdout) if not val \
                           else logging.NullHandler()
            if not val:
                self.handler.setFormatter(self.formatter)
            self.logger.addHandler(self.handler)

        super(FlorinLogger, self).__setattr__(key, val)

    def info(self, msg):
        self.logger.info(msg)

    def quiet(self):
        self.quiet = True

    def verbose(self):
        self.quiet = False


logger = FlorinLogger()
