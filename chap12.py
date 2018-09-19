# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "학습모드, 게임을 화면에서 보애주지 안습니다.")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 10000             # 최대 학습 횟수
TARGET_UPDATE_INTERVAL = 1000   # 학습을 일정 횟수만큼 진행할때마다 한번씩 목표 신경망 갱신
TRAIN_INTERVAL = 4              # 게임 4프레임(상태)마다 한번씩 학습
OBSERVE = 100                   # 일정 수준의 학습 데이터가 쌓이기 전에는 학습하지 않기

NUM_ACTION = 3                  # 행동 - 0: 좌, 1: 유지, 2: 우
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10


# 에이전트는 학습시키는 부분과 학습된 결과로 게임을 실행하는 부분 2개로 나뉨.
# 1. 학습부분
def train():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()
    game =Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
    # 최종 결과값 갯수 '선택할 행동의 갯수' NUM_ACTION 설정
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    # 학습결과 저장 및 확인
    # 한판마다 얻는 점수를 저장하고 확인
    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    # 파일 저장
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 목표신경망 초기화
    brain.update_target_network()

    # 행동을 선택할떄 DQN을 이용할 시점 정함
    # 일정시간이 지나기전에 행동을 무작위 선택하고 게임 진행중 epsilon값 줄여 나감
    epsilon = 1.0

    # 학습진행 조절을 위한 진행된 프레임 횟수
    time_step = 0
    # 학습결과를 확인하기 위한 점수 저장 배열
    total_reward_list = []

    # 학습 시작
    for episode in range(MAX_EPISODE):
        terminal = False            # 게임 종료
        total_reward = 0            # 한게임당 얻은 총 점수

        state = game.reset()        # 게임 초기화
        brain.init_state(state)     # DQN에 게임 초기화

        # 녹색사각형이 다른 사각형에 충돌할때까지 게임 수행
        while not terminal:

            # 학습 초반 (100회 이전)은 무작위로 수행
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            # 100회 이상이면 무작위값 사용비율을 줄여가면서 수행
            if episode > OBSERVE:
                epsilon -= 1 / 1000

            # 게임상태, 보상과 게임종료여부 받음
            state, reward, terminal = game.step(action)
            total_reward += reward

            # 현재상태를 신경망 객체에 기억
            # 기억된 정보를 이용하여 신경망 학습 시킴
            brain.remember(state, action, reward, terminal)

            # 프레임 100번이 넘으면 4프레임마다 한번씩 학습
            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()
            # 1000프레임 마다 한번씩 목표 신경망 갱신
            if time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()
            time_step += 1

        # 게임 종료시 획득점수 출력하고 점수 저장
        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))
        total_reward_list.append(total_reward)

        # 에피소드 10번마다 받은점수를 로그에 저장, 100마다 학습모델 저장
        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


# 게임 진행 부분
def replay():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game =Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    # 최종 결과값 갯수 '선택할 행동의 갯수' NUM_ACTION 설정
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    # 저장해둔 모델 읽어옴
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()

            state, reward, terminal = game.step(action)
            total_reward += reward
            brain.remember(state, action, reward, terminal)
            time.sleep(0.3)

        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))



def main(_):
    if FLAGS.train:
        train()
    else:
        replay()



if __name__ == '__main__':
    tf.app.run()
