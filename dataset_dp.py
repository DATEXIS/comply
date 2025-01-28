import torch
from tqdm import tqdm
import ray
import numpy as np
import itertools
import random

PRECISSION = 1.0e-30


class OWTDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        window_size,
        device,
        encoded_path,
        offsets_path,
        vocab_size,
        num_cpus,
        chunk_size,
        seed,
        max_batching_workers=40,  # noqa
        num_ddp_workers=-1,
        ddp_rank=-1,
        mock_data=False,
        mock_n_sentences=10_000_000,
        mock_sentence_length=200,
        mock_increment=50,  # noqa
        stride=1,
        data_subset=None,
    ):
        np.random.seed(seed)
        random.seed(seed)

        if not mock_data:
            encoded = np.load(encoded_path)  # [:1000_000]
            offsets = np.load(offsets_path)  # [:1_000_000]
            if data_subset is not None:
                encoded = np.load(encoded_path)[:data_subset]
                offsets = np.load(offsets_path)[:data_subset]
        else:
            # MAX_SENTENCES = int(400) # only one sequence per neuron
            MAX_SENTENCES = mock_n_sentences
            MAX_SENTENCE_LENGTH = mock_sentence_length
            INCREMENT = mock_increment
            seq_starts = np.random.randint(
                0, vocab_size - MAX_SENTENCE_LENGTH * INCREMENT, MAX_SENTENCES
            )
            seq_lens = (
                np.random.randint(
                    window_size, MAX_SENTENCE_LENGTH, MAX_SENTENCES
                )
                * INCREMENT
            )
            encoded = [
                np.arange(
                    seq_starts[i], seq_starts[i] + seq_lens[i], INCREMENT
                )
                for i in range(MAX_SENTENCES)
            ]  # noqa
            encoded = np.concatenate(encoded)
            offsets = np.cumsum(seq_lens // INCREMENT, dtype=np.int64)
        lengths = np.concatenate(
            (np.array((0,)).astype(np.int32), np.diff(offsets))
        )  # length of sentences
        usable_sentences = lengths > window_size
        self.samples = (lengths[usable_sentences] - window_size + 1).sum()
        self.total_tokens = len(encoded)
        self.total_sentences = len(offsets)

        self.encoded = ray.put(encoded.astype(np.int32))
        self.offsets = ray.put(offsets.astype(np.int32))
        self.lengths = ray.put(lengths.astype(np.int32))
        self.usable_sentences = ray.put(usable_sentences)
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.vocab_size = vocab_size
        self.num_cpus = num_cpus
        self.chunk_size = chunk_size
        self.max_batching_workers = max_batching_workers

        # MOCK DATA attributes
        self.mock_data = mock_data

        self.inv_P = ray.put(self.compute_inverse_frequencies())
        self.chunks, self.chunk_sizes = self.compute_chunks()

    def set_batch_size(self, batch_size):
        self.chunk_size = batch_size

    def compute_chunks(self):
        part_size = self.total_sentences // self.num_cpus
        part_indices = [
            (x, x + part_size)
            for x in range(0, self.total_sentences, part_size)
        ]
        part_indices[-1] = (part_indices[-1][0], self.total_sentences)
        futures = [
            self.compute_partof_chunks.remote(
                self.lengths,
                self.usable_sentences,
                self.chunk_size,
                self.window_size,
                start,
                end,
            )
            for (start, end) in part_indices
        ]  # noqa

        futures_order = [f for f in futures]
        chunks = [None] * len(part_indices)
        sizes = [None] * len(part_indices)
        with tqdm(
            total=len(part_indices),
            desc="Computing chunk indices for batching",
        ) as bar:  # noqa
            while len(futures) > 0:
                done_id, futures = ray.wait(futures)
                chunks[futures_order.index(done_id[0])] = ray.get(done_id[0])[
                    0
                ]
                sizes[futures_order.index(done_id[0])] = ray.get(done_id[0])[1]
                bar.update(1)

        def merge(x):
            return list(itertools.chain.from_iterable(x))

        return merge(chunks), merge(sizes)

    @ray.remote
    def compute_partof_chunks(
        lengths, usable_sentences, chunk_size, window_size, start, end
    ):  # noqa
        lengths = lengths[start:end]
        usable_sentences = usable_sentences[start:end]
        chunks, sizes = [], []
        running_size = chunk_size
        local_start = 0
        for i, s in enumerate(usable_sentences):
            if s:
                running_size -= lengths[i] - window_size + 1
            if running_size < 0:
                chunks.append((start + local_start, start + i + 1))
                sizes.append(chunk_size - running_size)  # Total sample size
                local_start = i + 1
                running_size = chunk_size
        return chunks, sizes

    def shuffle_chunks(self):
        chunks_sizes = list(zip(self.chunks, self.chunk_sizes))
        random.shuffle(chunks_sizes)
        self.chunks = [c[0] for c in chunks_sizes]
        self.chunk_sizes = [c[1] for c in chunks_sizes]

    @ray.remote
    def compute_word_frequency(encoded_id, vocab_size, chunk_start, chunk_end):
        P = np.zeros(vocab_size)
        chunk = encoded_id[chunk_start:chunk_end]

        token_ids, counts = np.unique(chunk, return_counts=True)
        for token_id, count in zip(token_ids, counts):
            P[token_id] = count
        return P

    # @profile
    def compute_inverse_frequencies(self):
        chunk_size = self.total_tokens // self.num_cpus
        chunk_indices = [
            (x, x + chunk_size)
            for x in range(0, self.total_tokens, chunk_size)
        ]
        futures = [
            self.compute_word_frequency.remote(
                self.encoded, self.vocab_size, start, end
            )
            for (start, end) in chunk_indices
        ]  # noqa

        futures_order = [f for f in futures]
        frequencies = [None] * len(chunk_indices)
        with tqdm(
            total=len(chunk_indices), desc="Computing Frequencies"
        ) as bar:  # noqa
            while len(futures) > 0:
                done_id, futures = ray.wait(futures)
                frequencies[futures_order.index(done_id[0])] = ray.get(
                    done_id[0]
                )
                bar.update(1)
        frequencies = np.stack(frequencies, axis=1).sum(axis=1)
        inv_frequencies = np.zeros_like(frequencies)
        for i, f in enumerate(frequencies):
            if f != 0.0:
                # The paper states inverse probabilities of the words thus:
                # inv_frequencies[i] = f/self.total_tokens
                # The cuda code shows inverse frequencies...???
                # inv_frequencies[i] = 1/(f/self.total_tokens)
                inv_frequencies[i] = 1 / f

        return inv_frequencies

    @ray.remote
    def build_batch(
        encoded,
        offsets,
        lengths,
        usable_sentences,
        window_size,
        batch_size,
        inv_P,
        start,
        end,
        mock_data,
        stride,
    ):  # noqa
        batch = []
        offsets = offsets[start:end]
        lengths = lengths[start:end]
        usable_sentences = usable_sentences[start:end]
        running_size = batch_size
        for i_sentence in range(end - start):
            if usable_sentences[i_sentence]:
                sentence = encoded[
                    (offsets[i_sentence] - lengths[i_sentence]) : offsets[
                        i_sentence
                    ]
                ]
                for w in range(0, len(sentence) - window_size + 1, stride):
                    if running_size > 0:
                        example_end = min(w + window_size, len(sentence) - 1)
                        if example_end - w >= window_size:
                            example = sentence[w:example_end]
                            positions = np.arange(w, example_end) / len(
                                sentence
                            )
                            batch += [
                                np.stack((example, inv_P[example], positions))
                            ]
                            running_size -= 1
        if len(batch) > 0:
            batch = np.stack(batch)
            np.random.shuffle(batch)
        else:
            batch = None
        return batch

    # @profile
    def __iter__(self):
        incomplete_batches = []
        intermediate_batch = None
        done_id = []
        futures = []
        consumed = 0
        # Guarantee that at most max_batching_workers are
        # alive building batches,
        # more will fill up the ram/disk
        concurrent_tasks = self.max_batching_workers
        for i, (start, end) in enumerate(self.chunks):
            if (
                len(futures) > concurrent_tasks
                or (len(self.chunks) - i) <= concurrent_tasks
            ):  # noqa
                total_ready = i - concurrent_tasks - consumed
                done_id, _ = ray.wait(futures, num_returns=total_ready)
                consumed += 1
                futures.pop(futures.index(done_id[0]))
                result = ray.get(done_id[0])
                if result is not None:
                    if len(result) == self.chunk_size:
                        yield result
                        del result
                    else:
                        incomplete_batches.append(result)
                        intermediate_batch = np.concatenate(incomplete_batches)
                        incomplete_batches = [intermediate_batch]
                        if len(intermediate_batch) > self.chunk_size:
                            ret = intermediate_batch[: self.chunk_size]
                            remaining = intermediate_batch[self.chunk_size :]
                            incomplete_batches = [remaining]
                            yield ret
                            del ret

            futures.append(
                self.build_batch.remote(
                    self.encoded,
                    self.offsets,
                    self.lengths,
                    self.usable_sentences,
                    self.window_size,
                    self.chunk_size,
                    self.inv_P,
                    start,
                    end,
                    self.mock_data,
                    self.stride,
                )
            )
        # Last futures
        while len(futures) > 0:
            done_id, futures = ray.wait(futures)
            result = ray.get(done_id[0])
            if result is None:
                continue

            if len(result) == self.chunk_size:
                yield result
            else:
                incomplete_batches.append(result)
                intermediate_batch = np.concatenate(incomplete_batches)
                incomplete_batches = [intermediate_batch]
                if len(intermediate_batch) > self.chunk_size:
                    ret = intermediate_batch[: self.chunk_size]
                    remaining = intermediate_batch[self.chunk_size :]
                    incomplete_batches = [remaining]
                    yield ret

        # last batches smaller than batch_size
        if len(incomplete_batches) == 1:
            yield incomplete_batches.pop()
