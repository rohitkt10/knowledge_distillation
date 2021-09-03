import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['Distiller']

class Distiller(tfk.Model):
	"""
	A base class for various distillation strategies.
	"""
	def __init__(self, student, teacher=None, precompute_teacher_logits=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if precompute_teacher_logits:
			self.precompute_teacher_logits = precompute_teacher_logits ## if precomputed logits
														## are available use that.
		else:
			assert teacher, "Teacher model not passed."
			self.teacher = teacher
		self.student = student

	def call(self, inputs, *args, **kwargs):
		return self.student.call(inputs, *args, **kwargs,)

	def save(self, *args, **kwargs):
		return self.student.save(*args, **kwargs)

	def save_weights(self, *args, **kwargs):
		return self.student.save_weights(*args, **kwargs)

	def train_step(self, *args, **kwargs):
		raise NotImplementedError("Implement this method depending on the distillation strategy.")
