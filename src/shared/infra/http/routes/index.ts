import { Router } from 'express';

import usersRouter from '@modules/users/infra/http/routes/users.routes';
import sessionsRouter from '@modules/users/infra/http/routes/sessions.routes';
import imagesRouter from '@modules/users/infra/http/routes/images.routes';
import petsRouter from '@modules/users/infra/http/routes/pets.routes';

const routes = Router();

routes.use('/users', usersRouter);
routes.use('/sessions', sessionsRouter);
routes.use('/images', imagesRouter);
routes.use('/pets', petsRouter);

export default routes;
