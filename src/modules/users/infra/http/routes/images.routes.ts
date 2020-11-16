import { Router } from 'express';
import multer from 'multer';
import uploadConfig from '@config/upload';

import ImagesController from '../controllers/ImagesController';
import ensureAutheticated from '../middlewares/ensureAuthenticated';

const imagesRouter = Router();
const upload = multer(uploadConfig.multer);
const imagesController = new ImagesController();

imagesRouter.get('/', ensureAutheticated, imagesController.index);

imagesRouter.post(
  '/',
  ensureAutheticated,
  upload.array('file'),
  imagesController.create,
);

imagesRouter.delete('/:case_id', ensureAutheticated, imagesController.delete);

export default imagesRouter;
