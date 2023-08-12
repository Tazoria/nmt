import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# import nmt.data_loader as data_loader
# from nmt.search import SingleBeamSearchBoard


# input: 인코더의 전체 타임스탭의 output, 디코더 현재 타임스탭의 output, 마스크
# output: contenxt vector
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        # hs크기만큼을 입력으로 받고 출력으로 내주기, bias는 필요 없음
        # w = Softmax(Q*W*K_t)
        # c = W * V
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    # h_t_tgt: Q -> 한타임스텝의 디코더의 히든 스테이트
    # h_src: 모든 타임스탭
    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length) - source sentence의 PAD 위치가 마스킹됨

        query = self.linear(h_t_tgt)
        # 입출력 사이즈 동일
        # |query| = (batch_size, 1, hidden_size)

        weight = torch.bmm(query, h_src.transpose(1, 2))
        # |K_t| -> h_src.transpose(1, 2) -> (batch_size, length, hidden_size) -> (batch_size, hidden_size, length)
        # |weight| = (batch_size, 1, hidden_size) * (batch_size, hidden_size, length) = (batch_size, 1, length)
        #   => (미니배치 내 각 샘플별, 디코더의 현재 타임스텝에 대해서, 인코더의 전체 타임스탭에 대한 웨이트값)
        if mask is not None:
            # mask 값이 1이면 weight를 음의 무한대로 설정해 softmax값을 0으로 만듦
            # masked_fill_(mask, value): PAD위치에 True(1)를 넣어놨을 것임. -> 1을 음의 무한대로 바꿔줌
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight, h_src)
        # |context_vector| = (batch_size, 1, hidden_size)
        #   => (미니배치 내 각 샘플별, 디코더의 현재 타임스텝에 대해서, 어텐션 결과로 얻어온 컨텍스트 백터값)

        return context_vector


# Embedding 텐서를 받아서 y, h를 리턴하도록 함 -> seq2seq에서 객체로 선언해서 활용하면 될 것
class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            word_vec_size,
            int(hidden_size/2),  # bi-directional LSTM(정방향, 역방향)
            num_layers=n_layers,
            dropout=dropout_p,  # LSTM 각 레이어 사이 들어갈 dropout
            bidirectional=True,  # bi-directional 설정
            batch_first=True
        )
        '''
        파이토치의 원래 rnn, LSTM의 입출력 텐서의 쉐입은 배치의 디맨젼이 첫번째가 아님
        다른 레이어들을 쓸 때는 항상 배치 디멘젼을 왼쪽에 먼저 놓기 때문에
        batch_first=True로 앞쪽으로 배치해줌
        '''

    def forward(self, emb):  # bi-directional이라 한 timestep씩 할 필요없이 전체 timestep이 인코더에 다 들어옴
        # |emb| = (batch_size, seq_length, word_vec_size)

        """
        임베딩 텐서가 텐서가 아니라 튜플인가?
            - 안해도 되긴 함
            - pack_padded_sequence(): 텐서를 padded_sequence를 이용해 PackedSequence객체로 변환됨
            - PackedSequence객체는 timestep 단위가 아니라 미니배치 단위의 정보를 담고 있음
            - 예시
                [ 패딩 넣기 ]
                    a = [torch.tensor([1,2,3]), torch.tensor([3,4])]  # 텐서인 리스트(?)
                    b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
                        # >>>>
                        # tensor([[ 1,  2,  3],
                        #     [ 3,  4,  0]])
                [ pack_padded_sequence() 적용 ]
                    torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
                    # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
                    # pack 결과 텐서와 배치사이즈가 같이 튜플로 리턴됨
                    # data: 각 timestep마다 pad를 제외하고 sequence를 나눠서 묶어줌(?)
                    #   => 다만 pad가 있는 timestep의 경우 패드가 어느 위치에 있었는지를 모르기 때문에 시퀀스 길이 순서대로 정렬해줘야함
                    #   => 토치 텍스트가 해줘서 미니배치 안에는 다 소팅돼서 나올 것임
                    # 배치사이즈: 텐서 안의 각 timestep의 샘플 갯수
        """
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)

        else:
            x = emb

        y, h = self.rnn(x)
        # y: 전체 timestep의 마지막 레이어의 hidden states
        # LSTM이기 때문에 h는 마지막 timestep의 (hidden state, cell state)의 튜플 형태
        # |y| = (batch_size, length, hidden_size) - hidden_size: 2로 나눈 값에 곱하기 2가 들어간 값(정방향, 역방향 나눠졌다가 결과적으로 더해짐)
        # 튜플이기 때문에 |h[0]| = (num_layers*2, batch_size, hidden_size/2)
        #   num_layers*2(레이어 수 X 방향 수)
        #   hidden_size(양방향이므로 나누기 2)
        #   h[1] = cell state

        # y를 packed sequence로 넣었으면 여전히 packed sequence -> unpack해주면 원하는 텐서사이즈로 돌아감
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,  # input feeding을 위함
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,  # uni-directional
            batch_first=True
        )

    #
    #
    '''
        - forward 함수 override
        - encoder의 forward와의 차이점: encoder에서는 모든 timestep이 한 번에 들어왔음
            => decoder의 forward: 한 timestep씩만 들어옴
                이유1. 추론: 모든 timestep을 모르기 때문에 한 timestep씩 들어와야 함
                이유2. 학습: input feeding때문에라도 한 timestep씩 들어와야 함
                    => 이전 timestep의 h tilde를 워드 임베딩 벡터와 함께 받아야 함
                    => 나올 때 까지 기다려야 하므로 인코더처럼 한번에 할 수 없음
    '''

    # 한 timestep의 임베딩이므로 인코더의 임베딩과 비교하기 위해 emb_t를 정의(emb subscript t)
    # h_t_1_tilde: t-1시점의 h tilde
    # h_t_1: t-1 시점의 hidden state(h_t_1, c_t_1)
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, word_vec_size)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)  # = shape[0]
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # 디코더의 첫 timestep인 경우 h_t_1_tilde 값 초기화
            # 원하는 실제 사이즈(batch_size, 1, hidden_size)를 0으로 채워줌
            # 텐서끼리 연산을 할 때 디바이스, 텐서 타입이 같아야 함
            # 직접 FloatTensor를 만들고 0으로 채운 후 디바이스로 보내줄 수도 있지만 가장 간단한 방법은
            #   emb_t와 같은 디바이스에 같은 타입으로 만들어주면 됨
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero()

        # Input feeding trick. - RNN에 넣기전에 붙여주기!
        # torch.cat([], dim): 리스트 안에 있는 원소들을 dim차원에서 붙여줌
        # |x| = (bs, 1, ws+hs)
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)

        y, h = self.rnn(x, h_t_1)

        return y, h
        # |y| = (bs, 1, hs)
        # |h[0]| = (n_layers, bs, hs)
        # 이후 출력값에 어텐션을 적용하고 컨텍스트 백터와 y를 concat해서 h_tilde를 구하고
        #   다음 timestep에 h_tilde와 generator의 출력값을 embedding vector에 넣어서
        #   word embedding과 같이 입력으로 들어옴


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # 입력으로 h_tilde가 들어옴
    # teacher forcing을 함으로써 h_t_1_tilde는 필요하지만 이전 timestep의 출력값은 필요 X
    # 학습할 때 모든 timestep을 한번에 통과시킴, 추론시에만 한 step씩
    def forward(self, x):
        # |x| = (batch_size, length, hidden_size) = h_tilde

        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)
        # (미니배치 내 각 샘플별, 각 timestep에 대한, 단어별 log확률분포)

        # log 확률을 반환함
        return y


class Seq2Seq(nn.Module):
    def __init__(
            self,
            input_size,
            word_vec_size,
            hidden_size,
            output_size,
            n_layers=4,
            dropout_p=.2
    ):
        self.input_size = input_size,
        self.word_vec_size = word_vec_size,
        self.hidden_size = hidden_size,
        self.output_size = output_size,
        self.n_layers = n_layers,
        self.dropout_p = dropout_p

        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size, hidden_size,
            n_layers=n_layers, dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size, hidden_size,
            n_layers, = n_layers, dropout_p = dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length-1 > 0:
                # 샘플 시퀀스 길이가 샘플 시퀀스 최대 길이보다 작을 때
                # pad 값들을 1로 만들어 어텐션 웨이트를 없앰
                # tensor.new(): PyTorch에서 주어진 tensor의 속성(데이터 타입, 디바이스 등)을 그대로 유지하면서 새로운 tensor를 생성하는 데 사용되는 메서드
                # new_zeros(), new_ones(), new_full()
                mask += [torch.cat([x.new_ones(1,l).zero_(),
                                    x.new_ones(1,(max_length-l))
                                    ], dim=-1)]
            else:
                # 샘플의 시퀀스 길이가 샘플 시퀀스 최대 길이와 같을 때
                # 마스크의 각 값들을 0으로 채움
                mask += [x.new_ones(1,l)].zero_()

            mask = torch.cat(mask, dim=0).bool()

            return mask


    def merge_encoder_hiddens(self, encoder_hiddens):
        new_hiddens = []
        new_cells = []

        # |hiddens| = (#layers*2, bs, hs)
        hiddens, cells = encoder_hiddens

        # 양방향 LSTM - i번째와 (i+1)번째 레이어는 반대 방향
        # 각 방향의 사이즈도 hidden size의 절반
        # 나중에 하나의 히든 레이어로 concat
        for i in range(0, hiddens.size(0), 2):
            new_hiddens += [torch.cat([hiddens[i], hiddens[i+1]], dim=-1)]  # dim=-1: 맨 끝 차원에 맞춰주셈
            new_cells += [torch.cat([cells[i], cells[i+1]], dim=-1)]

        new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)

        return (new_hiddens, new_cells)
    
    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        # 양방향을 한 방향으로 합침
        # 사이즈 변환 필요: (n_layers * 2, batch_size, hidden_size / 2) => (n_layers, batch_size, hidden_size)
        # 변환작업은 view()메소드 만으로는 이루어지지 않을 것
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        # |h_tgt|: (#layers*2, bs, hs/2)
        #   transpose: (bs, #layers*2. hs/2)
        #   view: (bs, #layers, hs)
        #   transpose: (#layers, bs, hs)
        # => 왜 이랬다 저랬다 함? view로 한번에 예쁘게 안바뀜(shape는 만들어지지만 내부 원소들 구성이 이상해짐)
        # contiguous(): 메모리상에 예쁘게 잘 붙어있게 선언해주기
        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()

        # merge_encoder_hiddens() 메소드를 사용해도됨 - 하지만 non-parallel way
        # h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        batch_size = tgt.size(0)

        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
            # |mask| = (batch_size, length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]

        # 입력 문장의 모든 timestep을 위한 word embedding vectors
        emb_src = self.emb_src(x)
        # |emb_src| = (batch_size, length, word_vec_size)

        # 인코더의 마지막 레이어의 히든스테이트가 디코더의 첫 히든 스테이트가 됨
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        # |h_0_tgt| = (n_layers * 2, batch_size, hidden_size / 2)

        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        emb_tgt = self.emb_dec(tgt)
        # |emb_tgt| = (batch_size, length, word_vec_size)
        # generator에서 한번에 입력받아서 log확률로 바꿔줄것임(?)
        h_tilde = []

        h_t_tilde = None  # 첫 hidden state값이라 None
        decoder_hidden = h_0_tgt  # 인코더의 마지막 hidden state값
        # 마지막 timestep이 끝날때까지 디코더 동작시키기
        for t in range(tgt.size(1)):
            # Teacher Forcing: take each input from training set,
            # not from the last time-step's output.
            # Because of Teacher Forcing,
            # training procedure and inference procedure becomes different.
            # Of course, because of sequential running in decoder,
            # this causes severe bottle-neck.
            # input feeding했기 때문에 디코더 자체는 한 타입스텝씩 돎
            # unsqueeze(1): 어차피 사실 emb_tgt[:,t:,:]는 (bs, (1, 날아가버림), ws)이지만 명시적으로 해주어 덜 헷갈리게!
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            # |emb_t| = (batch_size, 1, word_vec_size)
            # |h_t_tilde| = (batch_size, 1, hidden_size)

            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            # |decoder_output| = (batch_size, 1, hidden_size)
            # |decoder_hidden| = (n_layers, batch_size, hidden_size)

            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)

            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, context_vector], dim=-1)))

            # |h_t_tilde| = (batch_size, 1, hidden_size)

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        # |h_tilde| = (batch_size, length, hidden_size)

        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, length, output_size)

        return y_hat
