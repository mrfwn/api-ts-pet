import { Request, Response } from 'express';
import { classToClass } from 'class-transformer';
import { container } from 'tsyringe';
import CreateUserService from '@modules/users/services/CreateUserService';

export default class StudentsController {
  constructor() {}

  public async create(request: Request, response: Response): Promise<Response> {
    const createUser = container.resolve(CreateUserService);
    const { name, email, password } = request.body;
    const user = await createUser.execute({ name, email, password });
    return response.json(classToClass(user));
  }
}
