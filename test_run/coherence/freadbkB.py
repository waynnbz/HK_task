



def  freadbkB(infile, lines=1, bkformat='float32', r0=0, rN=0, c0=0, cN=0):
    '''

    :param infile:
    :param lines: number of lines in the file. Return matrix DATA of size (LINES, appropriate_cols)
    :param bkformat: reads in a format defined in FREAD, or a prepended 'cpx' for complex data types.
    :param r0, rN, c0, cN: cropping area
    :return:
    '''

    #TODO: File selecting GUI for no 'infile' input case

    # checking if while file should be read and set flag
    if c0==0 and r0==0:
        if rN==0 and cN==0:
            readwholefile = True
        if rN==lines:
            #TODO: unknown functions called to set 'bytesoerelem', 'filesize', and 'filewidth'
        if cN==filewidth:
            readwholefile = True

    if readwholefile: print(f'Reading whole file: {infile}')

    # Check bkformat for complex type: 'cpx*'
    if not bkformat.isalpha():
        raise Exception("")



    return data, count