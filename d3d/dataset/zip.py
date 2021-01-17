"""
This moduled provide patched version of builtin Zipfile class as in https://github.com/ThomasPinna/python_zipfile_improvement
You can have better speed when read several files from a zip file containing a large number of files
Modifications are marked with '===== PATCH ====='
"""
import io
import struct
from zipfile import *
from zipfile import (_CD_COMMENT_LENGTH, _CD_EXTRA_FIELD_LENGTH,
                     _CD_FILENAME_LENGTH, _CD_LOCAL_HEADER_OFFSET,
                     _CD_SIGNATURE, _ECD_COMMENT, _ECD_LOCATION, _ECD_OFFSET,
                     _ECD_SIGNATURE, _ECD_SIZE, _EndRecData, sizeCentralDir,
                     sizeEndCentDir64, sizeEndCentDir64Locator,
                     stringCentralDir, stringEndArchive64, structCentralDir, MAX_EXTRACT_VERSION)
from typing import Union, List

__all__ = ["PatchedZipFile"]

class PatchedZipFile(ZipFile):
    '''
    This class is based on build-in ZipFile class, which is further patched for better reading speed. The
    improvement is achieved by skip reading metadata of files not interested in.

    :param to_extract: specify the path (inside zip) of files to be extracted
    '''
    def __init__(self, file, mode="r", compression=ZIP_STORED, allowZip64=True,
                 to_extract: Union[List[str], str] = []):
        if not isinstance(to_extract, (list, tuple, set)):
            to_extract = [to_extract]
        self.to_extract = set(str(p) for p in to_extract)

        super().__init__(file=file, mode=mode,
                         compression=compression,
                         allowZip64=allowZip64)

    # patched implementation to reduce open time
    def _RealGetContents(self):
        """Read in the table of contents for the ZIP file."""
        fp = self.fp
        try:
            endrec = _EndRecData(fp)
        except OSError:
            raise BadZipFile("File is not a zip file")
        if not endrec:
            raise BadZipFile("File is not a zip file")
        if self.debug > 1:
            print(endrec)
        size_cd = endrec[_ECD_SIZE]             # bytes in central directory
        offset_cd = endrec[_ECD_OFFSET]         # offset of central directory
        self._comment = endrec[_ECD_COMMENT]    # archive comment

        # "concat" is zero, unless zip was concatenated to another file
        concat = endrec[_ECD_LOCATION] - size_cd - offset_cd
        if endrec[_ECD_SIGNATURE] == stringEndArchive64:
            # If Zip64 extension structures are present, account for them
            concat -= (sizeEndCentDir64 + sizeEndCentDir64Locator)

        if self.debug > 2:
            inferred = concat + offset_cd
            print("given, inferred, offset", offset_cd, inferred, concat)
        # self.start_dir:  Position of start of central directory
        self.start_dir = offset_cd + concat
        fp.seek(self.start_dir, 0)
        data = fp.read(size_cd)
        fp = io.BytesIO(data)
        total = 0
        while total < size_cd:
            centdir = fp.read(sizeCentralDir)
            if len(centdir) != sizeCentralDir:
                raise BadZipFile("Truncated central directory")
            centdir = struct.unpack(structCentralDir, centdir)
            if centdir[_CD_SIGNATURE] != stringCentralDir:
                raise BadZipFile("Bad magic number for central directory")
            if self.debug > 2:
                print(centdir)
            filename = fp.read(centdir[_CD_FILENAME_LENGTH])
            flags = centdir[5]
            if flags & 0x800:
                # UTF-8 file names extension
                filename = filename.decode('utf-8')
            else:
                # Historical ZIP filename encoding
                filename = filename.decode('cp437')
            # ===== PATCH =====
            if filename not in self.to_extract:
                fp.seek(centdir[_CD_EXTRA_FIELD_LENGTH] + centdir[_CD_COMMENT_LENGTH], 1)
                continue
            # =================
            # Create ZipInfo instance to store file information
            x = ZipInfo(filename)
            x.extra = fp.read(centdir[_CD_EXTRA_FIELD_LENGTH])
            x.comment = fp.read(centdir[_CD_COMMENT_LENGTH])
            x.header_offset = centdir[_CD_LOCAL_HEADER_OFFSET]
            (x.create_version, x.create_system, x.extract_version, x.reserved,
             x.flag_bits, x.compress_type, t, d,
             x.CRC, x.compress_size, x.file_size) = centdir[1:12]
            if x.extract_version > MAX_EXTRACT_VERSION:
                raise NotImplementedError("zip file version %.1f" %
                                          (x.extract_version / 10))
            x.volume, x.internal_attr, x.external_attr = centdir[15:18]
            # Convert date/time code to (year, month, day, hour, min, sec)
            x._raw_time = t
            x.date_time = ( (d>>9)+1980, (d>>5)&0xF, d&0x1F,
                            t>>11, (t>>5)&0x3F, (t&0x1F) * 2 )

            x._decodeExtra()
            x.header_offset = x.header_offset + concat
            self.filelist.append(x)
            self.NameToInfo[x.filename] = x

            # update total bytes read from central directory
            total = (total + sizeCentralDir + centdir[_CD_FILENAME_LENGTH]
                     + centdir[_CD_EXTRA_FIELD_LENGTH]
                     + centdir[_CD_COMMENT_LENGTH])

            if self.debug > 2:
                print("total", total)

            # ===== PATCH =====
            self.to_extract.remove(filename)
            if not len(self.to_extract):
                break
            # =================
