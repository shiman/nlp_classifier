class Codebook:
    def __init__(self, start=0, decode=False):
        """
        Initialize.
        :param start: the start point of the codes
        :param decode: whether you want to keep an extra list to decode
                       the index back to string
        """
        self._start = start
        self._index = start
        self._data = dict()
        self._decode = decode
        if decode:
            self._data_inv = range(start)

    def add(self, element):
        """
        Add an element to the code book.
        :param element: an element or a list of elements
        """
        if isinstance(element, list):
            for i in element:
                self.add(i)
        else:
            if element not in self._data:
                self._data[element] = self._index
                if self._decode:
                    self._data_inv.append(element)
                self._index += 1

    def __len__(self):
        return self._index - self._start

    def __contains__(self, item):
        return item in self._data

    def encode(self, item):
        """
        Convert an item into its code representation.
        :param item: item list or a single item
        :return: an indices list, an index, or None
        """
        if isinstance(item, list):
            return [self._data[x] for x in item if x in self._data]
        else:
            return self._data.get(item, None)

    def decode(self, index):
        """
        Convert an item index back into its string representation.
        """
        if not self._decode:
            raise Exception("Decoder not initialized")
        return self._data_inv[index]
