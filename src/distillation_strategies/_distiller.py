import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['Distiller']

class Distiller(tfk.Model):
	"""
	A base class for various distillation strategies.
	"""
	def __init__(self, student, teacher, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if teacher.trainable:
			teacher.trainable = False
		self.student = student
		self.teacher = teacher

	def call(self, inputs, *args, **kwargs):
		return self.student.call(inputs, *args, **kwargs,)

	def save(self, *args, **kwargs):
		return self.student.save(*args, **kwargs)

	def save_weights(self, *args, **kwargs):
		return self.student.save_weights(*args, **kwargs)

	def train_step(self, *args, **kwargs):
		raise NotImplementedError("Implement this method depending on the distillation strategy.")
