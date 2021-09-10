import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['Distiller']

class Distiller(tfk.Model):
	"""
	A base class for various distillation strategies.
	"""
	def __init__(self, student, teacher=None, precompute_teacher_logits=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if not precompute_teacher_logits:
			assert teacher, "Teacher model not passed."

		self.precompute_teacher_logits = False
		self._teacher = teacher
		self._teacher.trainable = False
		self._student = student

	@property
	def teacher(self):
		return self._teacher

	@property
	def student(self):
		return self._student

	@teacher.setter
	def teacher(self, model):
		self._teacher = model

	@student.setter
	def student(self, model):
		self._student = model

	def call(self, inputs, *args, **kwargs):
		return self.student.call(inputs, *args, **kwargs,)

	def save(self, *args, **kwargs):
		return self.student.save(*args, **kwargs)

	def save_weights(self, *args, **kwargs):
		return self.student.save_weights(*args, **kwargs)

	def train_step(self, *args, **kwargs):
		raise NotImplementedError("Implement this method depending on the distillation strategy.")
