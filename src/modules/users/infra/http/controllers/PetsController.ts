import { Request, Response } from 'express';
import { classToClass } from 'class-transformer';
import { container } from 'tsyringe';
import InsertPetService from '@modules/users/services/InsertPetService';

export default class CasesUserController {
  constructor() {}

  public async create(request: Request, response: Response): Promise<Response> {
    const insertPet = container.resolve(InsertPetService);
    const user_id = request.user.id;
    const imagesIds = request.body;
    const pet = await insertPet.execute({
      user_id,
      imagesIds,
    });
    return response.json(classToClass(pet));
  }
}
