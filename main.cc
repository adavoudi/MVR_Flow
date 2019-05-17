// Program to calculate factorial and
// multiplication of two numbers.
#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
}

AVFormatContext *fmt_ctx = NULL;
AVCodecContext *video_dec_ctx = NULL;
AVStream *video_stream = NULL;
const char *src_filename = NULL;

int video_stream_idx = -1;
AVFrame *frame = NULL;
AVFrame *frameRGB = NULL;
int video_frame_count = 0;
uint8_t *dst_data[4];
// static uint8_t         *buffer;
int dst_bufsize;
SwsContext *img_convert_ctx;
int dst_linesize[4];

void motionToColor(const cv::Mat &flow)
{

    cv::Mat dxdy[2]; //X,Y
    cv::split(flow, dxdy);
    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(dxdy[0], dxdy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat bgr; //CV_32FC3 matrix
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::resize(bgr, bgr, cv::Size(video_dec_ctx->width, video_dec_ctx->height));
    cv::imshow("optical flow", bgr);

    cv::waitKey(150);
}

void avframeToMat(const AVFrame *frame, cv::Mat &image)
{
    int width = frame->width;
    int height = frame->height;

    // Allocate the opencv mat and store its stride in a 1-element array
    if (image.rows != height || image.cols != width || image.type() != CV_8UC3)
        image = cv::Mat(height, width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = image.step1();

    // Convert the colour format and write directly to the opencv matrix
    SwsContext *conversion = sws_getContext(width, height, (AVPixelFormat)frame->format, width, height, AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
    sws_freeContext(conversion);
    // image = image.clone();
}

void ppm_save(AVFrame *incframe,
              char *filename)
{
    FILE *f;
    int xsize, ysize;

    xsize = incframe->width;
    ysize = incframe->height;

    sws_scale(img_convert_ctx,
              incframe->data,
              incframe->linesize,
              0,
              ysize,
              dst_data,
              dst_linesize);

    f = fopen(filename, "wb");

    fprintf(f,
            "P6\r\n%d %d\r\n%d\r\n",
            xsize,
            ysize,
            255);

    fwrite(dst_data[0],
           1,
           dst_bufsize,
           f);

    fclose(f);
}

std::pair<cv::Mat, cv::Mat> extractMVandResidual(const AVMotionVector *mvs, int size,
                                                 const cv::Mat &prevFrame, const cv::Mat &currFrame,
                                                 bool reset)
{
    static cv::Mat motionMat;
    static cv::Mat residual;

    const AVMotionVector *mv = &mvs[0];
    int motionMatCols = static_cast<int>(video_dec_ctx->width / mv->w);
    int motionMatRows = static_cast<int>(video_dec_ctx->height / mv->h);
    if (motionMat.rows != motionMatRows || motionMat.cols != motionMatCols)
    {
        motionMat = cv::Mat::zeros(motionMatRows, motionMatCols, CV_32FC2);
        residual = cv::Mat::zeros(motionMatRows, motionMatCols, CV_32FC3);
    }
    if (reset)
    {
        motionMat.setTo(cv::Scalar::all(0));
        residual.setTo(cv::Scalar::all(0));
    }

    cv::Mat motionCompensated = prevFrame.clone();

    for (int i = 0; i < size / sizeof(*mvs); i++)
    {
        const AVMotionVector *mv = &mvs[i];
        int x = static_cast<int>(mv->dst_x / mv->w);
        int y = static_cast<int>(mv->dst_y / mv->h);
        if (x < 0 || x >= motionMatCols || y < 0 || y >= motionMatRows ||
            (mv->dst_x == mv->src_x && mv->dst_y == mv->src_y) ||
            (mv->dst_x <0 || mv->dst_x >= currFrame.cols) || 
            (mv->dst_y <0 || mv->dst_y >= currFrame.rows))
            continue;
        float val_x = mv->dst_x - mv->src_x;
        float val_y = mv->dst_y - mv->src_y;

        motionMat.at<float>(y, x, 0) += val_x;
        motionMat.at<float>(y, x, 1) += val_y;

        int orig_dst_x = int(mv->dst_x / mv->w) * mv->w;
        int orig_dst_y = int(mv->dst_y / mv->h) * mv->h;
        int orig_src_x = int(mv->src_x / mv->w) * mv->w;
        int orig_src_y = int(mv->src_y / mv->h) * mv->h;

        int diff_x = mv->src_x - mv->dst_x;
        int diff_y = mv->src_y - mv->dst_y;

        int left = diff_x > 0 ? diff_x : 0;
        int right = diff_x > 0 ? mv->w : mv->w - diff_x;
        int top = diff_y > 0 ? diff_y : 0;
        int bottom = diff_y > 0 ? mv->h : mv->h - diff_y;

        cv::Rect fullImageRect = cv::Rect(cv::Point(0,0), prevFrame.size());
        cv::Rect srcRect = cv::Rect(
            cv::Point(right+orig_src_x, bottom + orig_src_y),
            cv::Point(left+orig_src_x, top + orig_src_y)) & fullImageRect;
        cv::Rect dstRect = cv::Rect(
            cv::Point(orig_dst_x+mv->w-left, orig_dst_y+mv->h-top),
            cv::Point(orig_dst_x, orig_dst_y)) & fullImageRect;

        if(srcRect.size() != dstRect.size() || 
            (srcRect.width == 0 || srcRect.height == 0) || 
            dstRect.width == 0 || dstRect.height == 0){
            continue;
        }
        
        prevFrame(srcRect).copyTo(motionCompensated(dstRect));
    }
    cv::absdiff(motionCompensated, currFrame, residual);
    cv::imshow("MotionComp", residual);
    // cv::waitKey(30);
    return std::pair<cv::Mat, cv::Mat>(motionMat, residual);
}

int decode_packet(const AVPacket *pkt)
{
    static cv::Mat image;
    char buf[1024];
    int ret = avcodec_send_packet(video_dec_ctx, pkt);
    if (ret < 0)
    {
        // fprintf(stderr, "Error while sending a packet to the decoder: %s\n", av_err2str(ret));
        return ret;
    }
    static cv::Mat prevFrame;
    static cv::Mat currFrame;
    bool reset = false;
    while (ret >= 0)
    {
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            break;
        }
        else if (ret < 0)
        {
            // fprintf(stderr, "Error while receiving a frame from the decoder: %s\n", av_err2str(ret));
            return ret;
        }

        if (ret >= 0)
        {
            int i;
            AVFrameSideData *sd;

            video_frame_count++;
            sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
            prevFrame = currFrame.clone();
            avframeToMat(frame, currFrame);
            if (sd && !prevFrame.empty())
            {
                const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                auto res = extractMVandResidual(mvs, sd->size, prevFrame, currFrame, reset);
                reset = false;
                // motionToColor(dxdy.first, dxdy.second);
            }
            else
            {
                reset = true;
            }
            std::cout << video_dec_ctx->frame_number << std::endl;
            snprintf(buf, sizeof(buf), "%s-%d.ppm", "images", video_dec_ctx->frame_number);
            // ppm_save(frame, buf);
            cv::imshow("image", currFrame);
            cv::waitKey(20);
            // }
            av_frame_unref(frame);
        }
    }

    return 0;
}

int open_codec_context(AVFormatContext *fmt_ctx, enum AVMediaType type)
{
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = NULL;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0);
    if (ret < 0)
    {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename);
        return ret;
    }
    else
    {
        int stream_idx = ret;
        st = fmt_ctx->streams[stream_idx];

        dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx)
        {
            fprintf(stderr, "Failed to allocate codec\n");
            return AVERROR(EINVAL);
        }

        ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to copy codec parameters to codec context\n");
            return ret;
        }

        /* Init the video decoder */
        av_dict_set(&opts, "flags2", "+export_mvs", 0);
        if ((ret = avcodec_open2(dec_ctx, dec, &opts)) < 0)
        {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
            return ret;
        }

        video_stream_idx = stream_idx;
        video_stream = fmt_ctx->streams[video_stream_idx];
        video_dec_ctx = dec_ctx;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int ret = 0;
    AVPacket pkt = {0};

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <video>\n", argv[0]);
        exit(1);
    }
    src_filename = argv[1];

    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0)
    {
        fprintf(stderr, "Could not open source file %s\n", src_filename);
        exit(1);
    }

    if (avformat_find_stream_info(fmt_ctx, NULL) < 0)
    {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO);

    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream)
    {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        ret = 1;
    }

    frameRGB = av_frame_alloc();
    frame = av_frame_alloc();
    if (!frame || !frameRGB)
    {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
    }

    printf("%d, %d\n", video_dec_ctx->width, video_dec_ctx->height);
    printf("framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags\n");

    int w = video_dec_ctx->width;
    int h = video_dec_ctx->height;

    dst_bufsize =
        av_image_alloc(dst_data,
                       dst_linesize,
                       w,
                       h,
                       AV_PIX_FMT_BGR24,
                       1);

    img_convert_ctx = sws_getContext(w, h,
                                     video_dec_ctx->pix_fmt,
                                     w, h, AV_PIX_FMT_BGR24, SWS_BICUBIC,
                                     NULL, NULL, NULL);

    if (img_convert_ctx == NULL)
    {
        fprintf(stderr, "Cannot initialize the conversion context!\n");
        exit(1);
    }

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0)
    {
        if (pkt.stream_index == video_stream_idx)
            ret = decode_packet(&pkt);
        av_packet_unref(&pkt);
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_packet(NULL);

end:
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    sws_freeContext(img_convert_ctx);
    return ret < 0;
}