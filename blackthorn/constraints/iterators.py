class ObjectIterator:
    """
    Class for iterating over properties of an object.
    """

    def __init__(self, base):
        """
        Create a new ObjectIterator that can iterate over properties of
        the input object.
        """
        self._base = base
        self._state = 0
        self._name = ""
        self._values = []

    def iterate(self, name, values):
        """
        Return an iterator over the property with the specified values.
        """
        self._state = 0
        self._name = name
        self._values = values
        return self

    def _model(self):
        val = self._values[self._state]
        self._base.__setattr__(self._name, val)
        return self._base

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        self._state = 0
        return self

    def __next__(self):
        if self._state < len(self._values):
            model = self._model()
            self._state += 1
            return model
        else:
            raise StopIteration


class ModelIterator:
    """
    Class for iterating over models.
    """

    def __init__(self, base):
        """
        Create a new ModelIterator for custom iteration over models.
        """
        self._base = base
        self._state = 0
        self._update = lambda model, _: model
        self._values = []

    def iterate(self, update, values):
        """
        Return an iterator over models where the base model is updated
        according to the update function.
        """
        self._state = 0
        self._update = update
        self._values = values
        return self

    def _model(self):
        val = self._values[self._state]
        self._update(self._base, val)
        return self._base

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        self._state = 0
        return self

    def __next__(self):
        if self._state < len(self._values):
            model = self._model()
            self._state += 1
            return model
        else:
            raise StopIteration
