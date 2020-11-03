import path from 'path';
import crypto from 'crypto';
import AppError from '@shared/errors/AppError';
import multer, { StorageEngine } from 'multer';
import { RequestHandler } from 'express';

const tmpFolder = path.resolve(__dirname, '..', '..', 'tmp');

const multerConfigStorage = {
  storage: multer.diskStorage({
    destination: tmpFolder,
    filename(request, file, callback) {
      const match = ['image/jpeg', 'image/png'];

      if (match.indexOf(file.mimetype) === -1) {
        throw new AppError(`${file.originalname} is invalid.`, 401);
      }

      const fileHash = crypto.randomBytes(10).toString('hex');
      const fileName = `${fileHash}-${file.originalname.replace(
        /[^A-Z0-9\\.]/gi,
        '_',
      )}`;
      return callback(null, fileName);
    },
  }),
  limits: { fileSize: 15000000 },
};

interface IUploadConfig {
  driver: 'disk';
  tmpFolder: string;
  uploadsFolder: string;
  multerArray: RequestHandler;
  multer: {
    storage: StorageEngine;
    limits: { fileSize: number };
  };
  config: {
    disk: {};
  };
}

export default {
  driver: process.env.STORAGE_DRIVER,
  tmpFolder,
  uploadsFolder: path.resolve(tmpFolder, 'uploads'),
  multer: multerConfigStorage,
  config: {
    disk: {},
  },
} as IUploadConfig;
