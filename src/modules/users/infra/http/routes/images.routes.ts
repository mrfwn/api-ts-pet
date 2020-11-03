import { Router } from 'express';
import multer from 'multer';
import uploadConfig from '@config/upload';

import ImagesController from '../controllers/ImagesController';
import ensureAutheticated from '../middlewares/ensureAuthenticated';

const imagesRouter = Router();
const upload = multer(uploadConfig.multer);
const casesStudentController = new ImagesController();

imagesRouter.get('/', ensureAutheticated, casesStudentController.index);

imagesRouter.post(
  '/',
  ensureAutheticated,
  upload.array('file'),
  casesStudentController.create,
);

imagesRouter.delete(
  '/:case_id',
  ensureAutheticated,
  casesStudentController.delete,
);

export default imagesRouter;
