import tensorflow as tf

global_step = tf.train.get_or_create_global_step()
schedule = [(1, 0.1), (2, 0.2)]
#schedule = sorted(config.lr_schedule.items(), key=lambda x: x[0])
num_train_iter = 10
boundaries = [num_train_iter*x[0] for x in schedule]
rates = [x[1] for x in schedule]
rates = rates[:1] + rates  # 
assert len(boundaries) + 1 == len(rates)

learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, rates)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #for ii in range(100):
    while True: 
      lr,gstep = sess.run([learning_rate,global_step])
      sess.run(global_step.assign(gstep+1))
      print(lr,gstep)
