#!/usr/bin/env python
# encoding: utf-8
"""
@author: fay
@contact: fayfeixiuhong
@time: 2020-12-29 11:31
@desc: # https://blog.csdn.net/weixin_30376163/article/details/94844046
"""

import tensorflow as tf
# feature_b = tf.feature_column.numeric_column("feature_b")
# feature_c_bucketized = tf.feature_column.bucketized_column(
# tf.feature_column.numeric_column("feature_c"),boundaries=[10,20])
# feature_a_x_feature_c = tf.feature_column.crossed_column(
#   columns=["feature_a", feature_c_bucketized])
#
# feature_columns = [feature_b, feature_c_bucketized, feature_a_x_feature_c]
# parsing_spec = tf.estimator.regressor_parse_example_spec(
#   feature_columns, label_key='my-label')

# For the above example, regressor_parse_example_spec would return the dict:


def _strip_leading_slashes(name):
    return name.rsplit('/', 1)[-1]

if __name__ == '__main__':
    # assert parsing_spec == {
    #     "feature_a": parsing_ops.VarLenFeature(tf.string),
    #     "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
    #     "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
    #     "my-label": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
    # }
    print(_strip_leading_slashes("/usr/ll/s"))


