from florin.pipelines.pipeline import Pipeline

class WorkQueuePipeline(Pipeline):
    def run(self, data):
        return next(map(self.operations, data))
