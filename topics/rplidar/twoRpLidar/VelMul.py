def velMul(posVelObss:list, velCoefficient:float):
    for counter,element in enumerate(posVelObss):
        posVelObss[counter][1] = velCoefficient * posVelObss[counter][1]
    return posVelObss