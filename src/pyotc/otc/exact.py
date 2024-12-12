from . import OTC

class ExactOTC(OTC):
    def step(self):
        raise NotImplementedError
        # TODO: implement exact step