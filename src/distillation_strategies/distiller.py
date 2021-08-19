import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

class BasicDistilledModel(tfk.Model):
	def __init__(
			self, 
			student, 
			teacher, 
			alpha=0.5,
			temperature=1.,
			name="distiller"):
		super().__init__(name=name)
		self.student = student
		self.teacher = teacher
		self.alpha = alpha
		self.temperature = temperature

	def compile(
			self,
			optimizer='rmsprop',
			student_loss=None,
			distillation_loss=None,
			augmentations=[],
			*args, **kwargs
				):
		super().compile(optimizer, *args, **kwargs)
		self.student_loss = student_loss
		self.distillation_loss = distillation_loss
		self.augmentations = augmentations

	def train_step(self, data):
		x, y = data

		# add augmented data 
		if len(self.augmentations) > 0:
			xs, ys = [x], [y]
			for aug in self.augmentations:
				_x, _y = aug(data)
				xs.append(_x)
				ys.append(_y)
			x, y = tf.concat(xs, axis=0), tf.concat(ys, axis=0)
		
		# teacher model predictions 
		y_pred_teacher = self.teacher(x, training=True)

		# record differentiable operations 
		with tf.GradientTape() as tape:
			y_pred_student = self.student(x, training=True)


			# compute losses 
			loss1 = self.student_loss(y, y_pred_student)
			loss2 = self.distillation_loss(
						tf.nn.softmax(y_pred_teacher/self.temperature),
						tf.nn.softmax(y_pred_student/self.temperature)
										)
			loss = self.alpha*loss1 + (1.-self.alpha)*loss2
			loss = loss + sum(self.student.losses)

		# compute gradients and update weights 
		trainable_variables = self.student.trainable_variables
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

		# update the metrics 
		self.compiled_metrics.update_state(y, y_pred_student)

		# return dictionary of metrics 
		return {m.name:m.result() for m in self.metrics}

	def test_step(self, data):
		x, y = data 
		y_pred = self.student(x, training=False)
		loss = self.student_loss(y, y_pred)
		self.compiled_metrics.update_state(y, y_pred)
		results = {m.name:m.result() for m in self.metrics}
		results.update({"student_loss":loss})
		return results




