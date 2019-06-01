__version__ = "0.6.2"

from gpt2sqa.gpt2.utils import load_tf_weights_in_gpt2
from gpt2sqa.gpt2.gpt2config import GPT2Config
from gpt2sqa.gpt2.gpt2model import GPT2Model
from gpt2sqa.gpt2.gptdoubleheads import GPT2DoubleHeadsModel
from gpt2sqa.gpt2.gpt2lmhead import GPT2LMHead


from .file_utils import PYTORCH_PRETRAINED_GPT2_CACHE, cached_path, WEIGHTS_NAME, CONFIG_NAME
