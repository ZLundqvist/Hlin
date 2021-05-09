from util.filesystem import ensure_cache_dir
from util.args import get_args
from pipeline.pipeline import Pipeline

args = get_args()
ensure_cache_dir()

pipeline = Pipeline(args)
pipeline.execute()

