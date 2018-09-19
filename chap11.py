# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

# 스크립트에서 받을 외부파라메터 기본값 설정
tf.app.flags.DEFINE_string("output_graph", "./workspace/flowers_graph.pb", "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels", "./workspace/flowers_labels.txt", "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image", True, "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS


def main(_):

    # 꽃이름 읽어 배열로 저장
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    # 꽃사진 예측파일을 읽어들여 신경망 그래프 생성 준비
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    # 읽어들인 신경망 모델에서 예측에 사용할 텐서 지정
    # 저장된 모델에서 최종 출력층은 'final_result:0'이라는 이름의 텐서 임.
    # 이 텐서를 가져와 예측에 사용
    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')
        # 예측 스크립트 실행 시 주어진 이름의 이미지파일 읽은 후 그 이미지를 예측모델에 넣어 예측 실행
        # 'DecodeJpeg/contents"0' 은 이미지 데이터를 입력값으로 넣을 플레이스홀더 이름
        image = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

    ######################################################################
    # 프로토콜 버퍼 형식으로 저장되어 있는 모델은 위와 같은 방법으로 사용 가능
    # 모델을 읽고 예측하는 코드

    print('=== 예측 결과 ===')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%)' % (name, score * 100))

    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    tf.app.run()
