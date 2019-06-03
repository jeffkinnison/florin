from florin.pipelines.pipeline import Pipeline

class BalsamPipeline(Pipeline):
    def run(self, data):
        return next(map(self.operations, data))
