import { Request, Response } from 'express';
import { classToClass } from 'class-transformer';
import { container } from 'tsyringe';

import ListImagesService from '@modules/users/services/ListImagesService';
import InsertImagesService from '@modules/users/services/InsertImagesService';
import DeleteImagesService from '@modules/users/services/DeleteImagesService';

export default class CasesUserController {
  constructor() {}

  public async index(request: Request, response: Response): Promise<Response> {
    const listAllImages = container.resolve(ListImagesService);
    const user_id = request.user.id;

    const casesStudent = await listAllImages.execute(user_id);

    return response.json(classToClass(casesStudent));
  }

  public async create(request: Request, response: Response): Promise<Response> {
    const insertImages = container.resolve(InsertImagesService);
    const user_id = request.user.id;
    const { files } = request;

    const filesName = files.map(file => file.filename);

    const images = await insertImages.execute({
      user_id,
      filesName,
    });

    return response.json(classToClass(images));
  }

  public async delete(request: Request, response: Response): Promise<Response> {
    const deleteImages = container.resolve(DeleteImagesService);
    const user_id = request.user.id;
    const { files } = request;
    const filesName = files.map(file => file.filename);
    await deleteImages.execute({ user_id, filesName });

    return response.json().status(200);
  }
}
