import { getRepository, Repository } from 'typeorm';
import IImagesRepository from '@modules/users/repositories/IImagesRepository';
import ICreateImageDTO from '@modules/users/dtos/ICreateImageDTO';
import Image from '../entities/Image';

class ImagesRepository implements IImagesRepository {
  private ormRepository: Repository<Image>;

  constructor() {
    this.ormRepository = getRepository(Image);
  }

  public async findAllImages(user_id: string): Promise<Image[]> {
    const listImages = await this.ormRepository.find({
      where: { user_id },
    });

    return listImages;
  }

  public async insertImages(fileList: ICreateImageDTO[]): Promise<Image[]> {
    const files = fileList.map(file => {
      return this.ormRepository.create(file);
    });

    return this.ormRepository.save(files);
  }

  public async insertPet(pet: ICreateImageDTO): Promise<Image> {
    const file = this.ormRepository.create(pet);

    return this.ormRepository.save(file);
  }

  public async deleteImages(filesName: string[]): Promise<void> {
    await Promise.all(
      filesName.map(async name => {
        const file = await this.ormRepository.findOne({
          select: ['id'],
          where: { name },
        });

        if (file) {
          await this.ormRepository.delete(file.id);
        }
      }),
    );
  }
}

export default ImagesRepository;
