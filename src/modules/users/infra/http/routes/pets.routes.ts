import { Router } from 'express';
import PetsController from '../controllers/PetsController';
import ensureAutheticated from '../middlewares/ensureAuthenticated';

const imagesRouter = Router();
const petsController = new PetsController();

imagesRouter.post('/', ensureAutheticated, petsController.create);

export default imagesRouter;
