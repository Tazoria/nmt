import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from nmt.utils import get_grad_norm, get_parameter_norm


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
        self.scaler = GradScaler()

    @staticmethod
    #@profile
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # AMP 시간~~
        # AMP(Automatic Mixed Precision): GPU의 한계로 인한 학습의 비효율성 해소
        # float 32 -> float 16 을 사용해 연산속도 높임
        with autocast(not engine.config.off_autocast):
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                # y_hat.size(-1) = output_size
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                # y도 bs로 쭉 펴주고 loss를 구함
                # view(-1): 모든 차원을 곱해 1차원으로 펴버림
                y.contiguous().view(-1)
            )
            # 미니배치 사이즈로 한번 나누고 거기서 또 interation 횟수만큼 나눠주기
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)

        # AMP의 예외처리: CPU인 경우 AMP X
        # cpu 시에는 스케일링 하지않고 backward()만, gpu시에는 스케일링 + backward()
        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        # g_norm은 안정적인 학습일 수록 점차 작아짐(점차 증가시 곧 발산)
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and engine.state.iteration > 0:
            torch_units.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            if engine.confing.gpu_id >= 0 and not engine.config.off_autocast:
                # 스케일링한 경우 바로 update
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

        loss = float(loss /word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        '''
            [ with torch.no_gard(): ]
            - PyTorch에서 그래디언트 계산을 비활성화하는 문맥(context)
            - 이 문맥 안에서의 모든 연산은 그래디언트 계산 X
            - 주로 모델의 추론(inference), 검증(validation) 단계에서 사용됨
        '''
        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                )
        word_count = int(mini_batch.tgt[1].sum())
        # 단어당 loss
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def attach(
            train_engine, validation_engine,
            training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
            validation_metric_names = ['loss', 'ppl'],
            verbose=VERBOSE_BATCH_WISE,
    ):

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x:x[metric_name]).attach(
                engine,
                metric_name,
            )
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss)
                ))
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # 마지막 epoch의 모델 파일이름 설정
        # 파일 이름에는 가능한 최대한의 정보를 넣을 것
        model_fn = config.model_fn.split('.')

        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss, np.exp(avg_train_loss)),
                                    '%.2f-%.2f' % (avg_valid_loss, np.exp(avg_valid_loss))] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # 현재 모델 저장
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )


class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(self,
              model, crit, optimizer,
              train_loader, valid_loader,
              src_vocab, tgt_vocab,
              n_epochs,
              lr_scheduler=None):
        # 학습 엔진과 검증 엔진 정의
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.vakudate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        # 학습 및 검증 엔진 과정의 출력 - Progress bar, metric
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # 매 학습 에폭마다 검증 1번 돌림
        # LR scheduler 필요시 적용 가능
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduuler is not None:
                engine.lr_scheduler.step()

            # 위의 콜백 함수들 attach
            train_engine.add_event_handler(
                Events.EPOCH_COMPLETED,
                run_validation,
                validation_engine,
                valid_loader
            )
            # 학습 시작을 위한 콜백 함수 attach
            train_engine.add_event_handler(
                Events.STARTED,
                self.target_engine_class.save_model,
                train_engine,
                self.config,
                src_vocab,
                tgt_vocab,
            )

            # 학습 시작
            train_engine.run(train_loader, max_epochs=n_epochs)

            return model
