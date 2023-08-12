import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

'''
[ ignite ]
  - 이벤트 기반으로 콜백 함수를 등록해서 원하는 작업들 등록
  - 이벤트가 트리거 됐을 때 콜백함수들 실행됨
  - ignite 엔진은 그냥 갖다쓰면 불편한 부분들이 있어서 상속받아서 활용함
  - train, validation 이 두개의 엔진이 필요함 => 이 두 엔진을 묶어주는 클래스도 필요함
  - process 함수를 받아서 엔진에 넣어주면 epoch에서 iteration마다 수행해줌
    - 원하는 프로시져를 구성해 training엔진, validation엔진 각각에 프로세스 함수로 등록해주기
  - train 함수는 ignite의 process function으로 등록이 되는데 process function은 인터페이스가 정해져있음
    - def train(엔진, 미니배치)
    - 나중에 train엔진을 run 시킬 때 데이터 로더(train, validation)를 각 엔진에 넣어줄 것임
    - 미니배치: train engine이 train 로더부터 미니배치를 받아 넣어줌
    - 엔진: 엔진 그 자체
'''

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

# from simple_nmt.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationEngine(Engine):

  def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
    self.model = model
    self.crit = crit
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.config = config

    super().__init__(func)

    self.best_loss = np.inf
    # AMP: with autocast를 해주고, scaler적용해줘야 함
    #   scaler 생성 해주기
    self.scaler = GradScaler()

  @staticmethod
  #@profile
  def trian(engine, mini_batch):
    # gradient descent를 한번에 처리해줄 것이기 때문에 그에 맞춰서 초기화 해주기
    engine.model.train()
    # engine.config.iteration_per_update: 업데이트(gd)를 몇번에 한번 할지에 대한 횟수로 밖에서 정의할 것임
      # engine.config.iteration_per_update == 1: 매번 업데이트 - 매번 zero_grad() 호출
    if engine.state.iteration % engine.config.iteration_per_update == 1 or engine.config.iteration_per_update == 1:
      if engine.state.iteration > 1:
        engine.optimizer.zero_grad()

    # lazy loading: GPU로 데이터를 한번에 적재해놓으면 빠르지만 메모리 사용량이 큼을 해결하기 위함
    #   => lazy loading 한다고 해서 속도상의 손실이 크지 않아서 lazy loading을 선택
    #   => 데이터 로더는 다른 곳에서 선언했고, 현재 모델의 device를 구해 미니배치의 각 텐서들을 해당 디바이스로 보내주기만 하면 됨
    # 모델의 device를 구해오는 방법: model.device 속성이 없음
    #   => model.parameters()를 호출해서 next()를 취하면 첫번째 파라미터가 나올 것임 - 그 파라미터의 device속성이 현재 모델의 device가 됨
    device = next(engine.model.parameters()).device
    # mini_batch: trainer가 리턴하는 객체가 될 것
    # 두 변수는 torch.text가 제공하는 형태로 돼있음
    # mini_batch.src[0].to(device): 실제 텐서
    # mini_batch.src[1]: length
    mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
    mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

    # Raw target variable has both BOS and EOS token.
    # The output of sequence-to-sequence does not have BOS token.
    # Thus, remove BOS token for reference.
    # x: 인코더에는 source의 전체(해당 텐서, length)를 넣어줌
    # y: decoder에 들어가는 target y의 경우에는 length텐서 빼고 해당 문장의 텐서만 넣어주면 됨
    #   => [:, 1:]: mini_batch.tgt의 <BOS>, <EOS>때문
    #   => 모델에 넣어주는 입력값과 함께 들어가는 <BOS>가 포함됨
    #   => 예측과 정답을 비교하기 위해 loss를 구할 때는 <EOS>가 포함됨

    x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
    # |x| = (batch_size, length)
    # |y| = (batch_size, length)

    with autocast(not engine.config.off_autocast):
      #
      y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])

      loss = engine.crit(
        y_hat.contiguous().view(-1, y_hat.size(-1)),
        y.contiguous().view(-1)
      )
      backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
