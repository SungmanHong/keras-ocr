# pylint: disable=invalid-name,too-many-locals,too-many-arguments
import typing
import string

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

from . import tools

DEFAULT_BUILD_PARAMS = {
    'height': 31,
    'width': 200,
    'color': False,
    'filters': (64, 128, 256, 256, 512, 512, 512),
    'rnn_units': (128, 128),
    'dropout': 0.25,
    'rnn_steps_to_discard': 2,
    'pool_size': 2,
    'stn': True,
}

DEFAULT_ALPHABET = string.digits + string.ascii_letters + '!\"#$%&()*+,-./:;?@[\\]^_`{|}~'+'가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝'


PRETRAINED_WEIGHTS = {
    'kurapan': {
        'alphabet': DEFAULT_ALPHABET,
        'build_params': DEFAULT_BUILD_PARAMS,
        'weights': {
            'notop': {
                'url': 'https://www.mediafire.com/file/n9yfn5wueu82rgf/crnn_kurapan_notop.h5/file',
                'filename': 'crnn_kurapan_notop.h5',
                'sha256': '027fd2cced3cbea0c4f5894bb8e9e85bac04f11daf96b8fdcf1e4ee95dcf51b9'
            },
            'top': {
                'url': 'https://www.mediafire.com/file/pkj2p29b1f6fpil/crnn_kurapan.h5/file',
                'filename': 'crnn_kurapan.h5',
                'sha256': 'a7d8086ac8f5c3d6a0a828f7d6fbabcaf815415dd125c32533013f85603be46d'
            }
        }
    }
}


def swish(x, beta=1):
    return x * keras.backend.sigmoid(beta * x)


keras.utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})


def _repeat(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])


def _meshgrid(height, width):
    x_linspace = tf.linspace(-1., 1., width)
    y_linspace = tf.linspace(-1., 1., height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
    y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid


# pylint: disable=too-many-statements
def _transform(inputs):
    locnet_x, locnet_y = inputs
    output_size = locnet_x.shape[1:]
    batch_size = tf.shape(locnet_x)[0]
    height = tf.shape(locnet_x)[1]
    width = tf.shape(locnet_x)[2]
    num_channels = tf.shape(locnet_x)[3]

    locnet_y = tf.reshape(locnet_y, shape=(batch_size, 2, 3))

    locnet_y = tf.reshape(locnet_y, (-1, 2, 3))
    locnet_y = tf.cast(locnet_y, 'float32')

    output_height = output_size[0]
    output_width = output_size[1]
    indices_grid = _meshgrid(output_height, output_width)
    indices_grid = tf.expand_dims(indices_grid, 0)
    indices_grid = tf.reshape(indices_grid, [-1])  # flatten?
    indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

    transformed_grid = tf.matmul(locnet_y, indices_grid)
    x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    x = tf.reshape(x_s, [-1])
    y = tf.reshape(y_s, [-1])

    # Interpolate
    height_float = tf.cast(height, dtype='float32')
    width_float = tf.cast(width, dtype='float32')

    output_height = output_size[0]
    output_width = output_size[1]

    x = tf.cast(x, dtype='float32')
    y = tf.cast(y, dtype='float32')
    x = .5 * (x + 1.0) * width_float
    y = .5 * (y + 1.0) * height_float

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    max_y = tf.cast(height - 1, dtype='int32')
    max_x = tf.cast(width - 1, dtype='int32')
    zero = tf.zeros([], dtype='int32')

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    flat_image_dimensions = width * height
    pixels_batch = tf.range(batch_size) * flat_image_dimensions
    flat_output_dimensions = output_height * output_width
    base = _repeat(pixels_batch, flat_output_dimensions)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    indices_a = base_y0 + x0
    indices_b = base_y1 + x0
    indices_c = base_y0 + x1
    indices_d = base_y1 + x1

    flat_image = tf.reshape(locnet_x, shape=(-1, num_channels))
    flat_image = tf.cast(flat_image, dtype='float32')
    pixel_values_a = tf.gather(flat_image, indices_a)
    pixel_values_b = tf.gather(flat_image, indices_b)
    pixel_values_c = tf.gather(flat_image, indices_c)
    pixel_values_d = tf.gather(flat_image, indices_d)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
    area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
    area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
    area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
    transformed_image = tf.add_n([
        area_a * pixel_values_a, area_b * pixel_values_b, area_c * pixel_values_c,
        area_d * pixel_values_d
    ])
    # Finished interpolation

    transformed_image = tf.reshape(transformed_image,
                                   shape=(batch_size, output_height, output_width, num_channels))
    return transformed_image


def CTCDecoder():
    def decoder(y_pred):
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
            input_shape[1], 'float32')
        unpadded = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
        unpadded_shape = tf.keras.backend.shape(unpadded)
        padded = tf.pad(unpadded,
                        paddings=[[0, 0], [0, input_shape[1] - unpadded_shape[1]]],
                        constant_values=-1)
        return padded

    return tf.keras.layers.Lambda(decoder, name='decode')


def build_model(alphabet,
                height,
                width,
                color,
                filters,
                rnn_units,
                dropout,
                rnn_steps_to_discard,
                pool_size,
                stn=True):
    """Build a Keras CRNN model for character recognition.

    Args:
        height: The height of cropped images
        width: The width of cropped images
        color: Whether the inputs should be in color (RGB)
        filters: The number of filters to use for each of the 7 convolutional layers
        rnn_units: The number of units for each of the RNN layers
        dropout: The dropout to use for the final layer
        rnn_steps_to_discard: The number of initial RNN steps to discard
        pool_size: The size of the pooling steps
        stn: Whether to add a Spatial Transformer layer
    """
    assert len(filters) == 7, '7 CNN filters must be provided.'
    assert len(rnn_units) == 2, '2 RNN filters must be provided.'
    inputs = keras.layers.Input((height, width, 3 if color else 1))
    x = keras.layers.Permute((2, 1, 3))(inputs)
    x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_3')(x)
    x = keras.layers.Conv2D(filters[3], (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = keras.layers.Conv2D(filters[4], (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_5')(x)
    x = keras.layers.Conv2D(filters[5], (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = keras.layers.Conv2D(filters[6], (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_7')(x)
    if stn:
        # pylint: disable=pointless-string-statement
        """Spatial Transformer Layer
        Implements a spatial transformer layer as described in [1]_.
        Borrowed from [2]_:
        downsample_fator : float
            A value of 1 will keep the orignal size of the image.
            Values larger than 1 will down sample the image. Values below 1 will
            upsample the image.
            example image: height= 100, width = 200
            downsample_factor = 2
            output image will then be 50, 100
        References
        ----------
        .. [1]  Spatial Transformer Networks
                Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
                Submitted on 5 Jun 2015
        .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
        .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
        """
        stn_input_output_shape = (width // pool_size**2, height // pool_size**2, filters[6])
        stn_input_layer = keras.layers.Input(shape=stn_input_output_shape)
        locnet_y = keras.layers.Conv2D(16, (5, 5), padding='same',
                                       activation='relu')(stn_input_layer)
        locnet_y = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(locnet_y)
        locnet_y = keras.layers.Flatten()(locnet_y)
        locnet_y = keras.layers.Dense(64, activation='relu')(locnet_y)
        locnet_y = keras.layers.Dense(6,
                                      weights=[
                                          np.zeros((64, 6), dtype='float32'),
                                          np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
                                      ])(locnet_y)
        localization_net = keras.models.Model(inputs=stn_input_layer, outputs=locnet_y)
        x = keras.layers.Lambda(_transform,
                                output_shape=stn_input_output_shape)([x, localization_net(x)])
    x = keras.layers.Reshape(target_shape=(width // pool_size**2,
                                           (height // pool_size**2) * filters[-1]),
                             name='reshape')(x)

    x = keras.layers.Dense(rnn_units[0], activation='relu', name='fc_9')(x)

    rnn_1_forward = keras.layers.LSTM(rnn_units[0],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_10')(x)
    rnn_1_back = keras.layers.LSTM(rnn_units[0],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_10_back')(x)
    rnn_1_add = keras.layers.Add()([rnn_1_forward, rnn_1_back])
    rnn_2_forward = keras.layers.LSTM(rnn_units[1],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_11')(rnn_1_add)
    rnn_2_back = keras.layers.LSTM(rnn_units[1],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_11_back')(rnn_1_add)
    x = keras.layers.Concatenate()([rnn_2_forward, rnn_2_back])
    backbone = keras.models.Model(inputs=inputs, outputs=x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    x = keras.layers.Dense(len(alphabet) + 1,
                           kernel_initializer='he_normal',
                           activation='softmax',
                           name='fc_12')(x)
    x = keras.layers.Lambda(lambda x: x[:, rnn_steps_to_discard:])(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    prediction_model = keras.models.Model(inputs=inputs, outputs=CTCDecoder()(model.output))
    labels = keras.layers.Input(name='labels', shape=[model.output_shape[1]], dtype='float32')
    label_length = keras.layers.Input(shape=[1])
    input_length = keras.layers.Input(shape=[1])
    loss = keras.layers.Lambda(lambda inputs: keras.backend.ctc_batch_cost(
        y_true=inputs[0], y_pred=inputs[1], input_length=inputs[2], label_length=inputs[3]))(
            [labels, model.output, input_length, label_length])
    training_model = keras.models.Model(inputs=[model.input, labels, input_length, label_length],
                                        outputs=loss)
    return backbone, model, training_model, prediction_model


class Recognizer:
    """A text detector using the CRNN architecture.

    Args:
        alphabet: The alphabet the model should recognize.
        build_params: A dictionary of build parameters for the model.
            See `keras_ocr.recognition.build_model` for details.
        weights: The starting weight configuration for the model.
        include_top: Whether to include the final classification layer in the model (set
            to False to use a custom alphabet).
    """
    def __init__(self, alphabet=None, weights='kurapan', build_params=None):
        assert alphabet or weights, 'At least one of alphabet or weights must be provided.'
        if weights is not None:
            build_params = build_params or PRETRAINED_WEIGHTS[weights]['build_params']
            alphabet = alphabet or PRETRAINED_WEIGHTS[weights]['alphabet']
        build_params = build_params or DEFAULT_BUILD_PARAMS
        if alphabet is None:
            alphabet = DEFAULT_ALPHABET
        self.alphabet = alphabet
        self.blank_label_idx = len(alphabet)
        self.backbone, self.model, self.training_model, self.prediction_model = build_model(
            alphabet=alphabet, **build_params)
        if weights is not None:
            weights_dict = PRETRAINED_WEIGHTS[weights]
            if alphabet == weights_dict['alphabet']:
                self.model.load_weights(
                    tools.download_and_verify(url=weights_dict['weights']['top']['url'],
                                              filename=weights_dict['weights']['top']['filename'],
                                              sha256=weights_dict['weights']['top']['sha256']))
            else:
                print('Provided alphabet does not match pretrained alphabet. '
                      'Using backbone weights only.')
                self.backbone.load_weights(
                    tools.download_and_verify(url=weights_dict['weights']['notop']['url'],
                                              filename=weights_dict['weights']['notop']['filename'],
                                              sha256=weights_dict['weights']['notop']['sha256']))

    def get_batch_generator(self, image_generator, batch_size=8, lowercase=False):
        """
        Generate batches of training data from an image generator. The generator
        should yield tuples of (image, sentence) where image contains a single
        line of text and sentence is a string representing the contents of
        the image. If a sample weight is desired, it can be provided as a third
        entry in the tuple, making each tuple an (image, sentence, weight) tuple.

        Args:
            image_generator: An image / sentence tuple generator. The images should
                be in color even if the OCR is setup to handle grayscale as they
                will be converted here.
            batch_size: How many images to generate at a time.
            lowercase: Whether to convert all characters to lowercase before
                encoding.
        """
        y = np.zeros((batch_size, 1))
        if self.training_model is None:
            raise Exception('You must first call create_training_model().')
        max_string_length = self.training_model.input_shape[1][1]
        while True:
            batch = [sample for sample, _ in zip(image_generator, range(batch_size))]
            if not self.model.input_shape[-1] == 3:
                images = [
                    cv2.cvtColor(sample[0], cv2.COLOR_RGB2GRAY)[..., np.newaxis] for sample in batch
                ]
            else:
                images = [sample[0] for sample in batch]
            images = np.array([image.astype('float32') / 255 for image in images])
            sentences = [sample[1].strip() for sample in batch]
            if lowercase:
                sentences = [sentence.lower() for sentence in sentences]
            
            
            assert all(c in self.alphabet
                for c in ''.join(sentences)), 'Found illegal characters in sentence.{0},{1}'.format(self.alphabet, sentences)
            assert all(sentences), 'Found a zero length sentence. SENTENCE:%s' %sentences
            assert all(
                len(sentence) <= max_string_length
                for sentence in sentences), 'A sentence is longer than this model can predict. SENTENCE:%s' %sentences
            assert all("  " not in sentence for sentence in sentences), (
                'Strings with multiple sequential spaces are not permitted. '
                'See https://github.com/faustomorales/keras-ocr/issues/54')

            
            
            label_length = np.array([len(sentence) for sentence in sentences])[:, np.newaxis]
            labels = np.array([[self.alphabet.index(c)
                                for c in sentence] + [-1] * (max_string_length - len(sentence))
                               for sentence in sentences])
            input_length = np.ones((batch_size, 1)) * max_string_length
            if len(batch[0]) == 3:
                sample_weights = np.array([sample[2] for sample in batch])
                yield (images, labels, input_length, label_length), y, sample_weights
            else:
                yield (images, labels, input_length, label_length), y

    def recognize(self, image):
        """Recognize text from a single image.

        Args:
            image: A pre-cropped image containing characters
        """
        image = tools.read_and_fit(filepath_or_array=image,
                                   width=self.prediction_model.input_shape[2],
                                   height=self.prediction_model.input_shape[1],
                                   cval=0)
        if self.prediction_model.input_shape[-1] == 1 and image.shape[-1] == 3:
            # Convert color to grayscale
            image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        image = image.astype('float32') / 255
        return ''.join([
            self.alphabet[idx] for idx in self.prediction_model.predict(image[np.newaxis])[0]
            if idx not in [self.blank_label_idx, -1]
        ])

    def recognize_from_boxes(self, images, box_groups, **kwargs) -> typing.List[str]:
        """Recognize text from images using lists of bounding boxes.

        Args:
            images: A list of input images, supplied as numpy arrays with shape
                (H, W, 3).
            boxes: A list of groups of boxes, one for each image
        """
        assert len(box_groups) == len(images), \
            'You must provide the same number of box groups as images.'
        crops = []
        start_end = []
        for image, boxes in zip(images, box_groups):
            image = tools.read(image)
            if self.prediction_model.input_shape[-1] == 1 and image.shape[-1] == 3:
                # Convert color to grayscale
                image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
            for box in boxes:
                crops.append(
                    tools.warpBox(image=image,
                                  box=box,
                                  target_height=self.model.input_shape[1],
                                  target_width=self.model.input_shape[2]))
            start = 0 if not start_end else start_end[-1][1]
            start_end.append((start, start + len(boxes)))
        if not crops:
            return [[] for image in images]
        X = np.float32(crops) / 255
        if len(X.shape) == 3:
            X = X[..., np.newaxis]
        predictions = [
            ''.join([self.alphabet[idx] for idx in row if idx not in [self.blank_label_idx, -1]])
            for row in self.prediction_model.predict(X, **kwargs)
        ]
        return [predictions[start:end] for start, end in start_end]

    def compile(self, *args, **kwargs):
        """Compile the training model."""
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = 'RMSprop'
        if 'loss' not in kwargs:
            kwargs['loss'] = lambda _, y_pred: y_pred
        self.training_model.compile(*args, **kwargs)
