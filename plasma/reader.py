import vcf
from vcf.model import _Record

class SubsetReader(vcf.Reader):
	def __init__(self, **kwargs):
		self.snp_subset = kwargs.pop("snp_subset")
		self.sample_subset = kwargs.pop("sample_subset")
		super().__init__(**kwargs)

	def _parse_metainfo(self):
		super()._parse_metainfo()
		self.sample_pos = [ind for ind, val in enumerate(self.samples) if val in self.sample_subset]
		self.samples = [i for i in self.samples if i in self.sample_subset]
		self._sample_indexes = {k: v for k, v in self._sample_indexes.items() if k in self.sample_subset}

	def _parse_samples(self, samples, samp_fmt, site):
		samples_sel = [samples[i] for i in self.sample_pos]
		return super()._parse_samples(samples_sel, samp_fmt, site)

	def _parse_filter(self, filt_str):
		if filt_str == '.':
			return None
		elif filt_str == 'PASS':
			return []
		else:
			return filt_str.split(';')

	def __next__(self):
		'''Return the next record in the file.'''
		while True:
			line = next(self.reader)
			row = self._row_pattern.split(line.rstrip())

			if row[2] != '.':
				ID = row[2]
			else:
				ID = None

			if ID in self.snp_subset:
				break

		chrom = row[0]
		if self._prepend_chr:
			chrom = 'chr' + chrom
		pos = int(row[1])

		ref = row[3]
		alt = self._map(self._parse_alt, row[4].split(','))

		try:
			qual = int(row[5])
		except ValueError:
			try:
				qual = float(row[5])
			except ValueError:
				qual = None

		filt = self._parse_filter(row[6])
		info = self._parse_info(row[7])

		try:
			fmt = row[8]
		except IndexError:
			fmt = None
		else:
			if fmt == '.':
				fmt = None

		record = _Record(chrom, pos, ID, ref, alt, qual, filt,
				info, fmt, self._sample_indexes)

		if fmt is not None:
			samples = self._parse_samples(row[9:], fmt, record)
			record.samples = samples

		return record