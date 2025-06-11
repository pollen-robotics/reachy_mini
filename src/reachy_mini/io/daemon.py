from reachy_mini import MujocoBackend
from reachy_mini.io import Server


class Daemon:
    def __init__(self, sim=False):
        # TODO handle scene selection for mujoco backend
        if sim:
            self.backend = MujocoBackend()
        else:
            raise NotImplementedError("Real robot backend is not implemented yet.")

        self.server = Server(self.backend, localhost_only=False)
        self.server.start()

    def run(self):
        self.backend.run()


if __name__ == "__main__":
    d = Daemon(sim=True)
    d.run()
