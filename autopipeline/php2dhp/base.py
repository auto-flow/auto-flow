class PHP2DHP():
    def convert(self, php):
        raise NotImplementedError()

    def __call__(self, php):
        return self.convert(php)
