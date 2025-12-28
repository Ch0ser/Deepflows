from numpy.random import permutation
from .dataset import Dataset


class Sampler:
    def __init__(self, dataset):
        pass

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class RandomSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        yield from permutation(len(self.dataset)).tolist()

    def __len__(self):
        return len(self.dataset)


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _DataLoaderIter:
    def __init__(self, loader):
        self.loader = loader
        self.sample_iter = iter(self.loader.batch_sampler)
        self.buffer = []
        self._fill_buffer()

    def _fill_buffer(self):
        while len(self.buffer) < self.loader.prefetch_size:
            try:
                index = next(self.sample_iter)
            except StopIteration:
                break
            batch = self.loader.dataset[index]
            if self.loader.as_contiguous:
                x, y = batch
                try:
                    import numpy as np
                    x = np.ascontiguousarray(x)
                    y = np.ascontiguousarray(y)
                    batch = (x, y)
                except Exception:
                    pass
            self.buffer.append(batch)

    def __next__(self):
        if self.buffer:
            batch = self.buffer.pop(0)
            self._fill_buffer()
            return batch
        index = next(self.sample_iter)
        batch = self.loader.dataset[index]
        if self.loader.as_contiguous:
            import numpy as np
            batch = (np.ascontiguousarray(batch[0]), np.ascontiguousarray(batch[1]))
        return batch


class DataLoader:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 prefetch_size: int = 0,
                 as_contiguous: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_size = max(0, int(prefetch_size))
        self.as_contiguous = as_contiguous

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return _DataLoaderIter(self)


def data_loader(X, y, batch_size, shuffle=False, prefetch_size: int = 0, as_contiguous: bool = True):
    class TrainSet(Dataset):
        def __init__(self, X, y):
            super().__init__()
            self.data = X
            self.target = y

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return len(self.data)

    return DataLoader(TrainSet(X, y), batch_size, shuffle, prefetch_size=prefetch_size, as_contiguous=as_contiguous)
