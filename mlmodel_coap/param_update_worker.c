/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 * 
 * Adapted from RIOT/sys/suit/transport/worker.c
 * 
 */

/*
 * Copyright (C) 2019 Freie Universität Berlin
 *               2019 Inria
 *               2019 Kaspar Schleiser <kaspar@schleiser.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v2.1. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     sys_suit
 * @{
 *
 * @file
 * @brief       SUIT transport worker thread
 *
 * @author      Koen Zandberg <koen@bergzand.net>
 * @author      Kaspar Schleiser <kaspar@schleiser.de>
 * @author      Francisco Molina <francois-xavier.molina@inria.fr>
 * @author      Alexandre Abadie <alexandre.abadie@inria.fr>
 * @}
 */

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>
#include <sys/types.h>

#include "mutex.h"
#include "log.h"
#include "thread.h"

#include "suit/transport/worker.h"

// #ifdef MODULE_SUIT_TRANSPORT_COAP
#include "net/nanocoap_sock.h"
#include "suit/transport/coap.h"
#include "net/sock/util.h"
// #endif


// #ifdef MODULE_SUIT
#include "suit.h"
#include "suit/handlers.h"
#include "suit/storage.h"
// #endif

#define ENABLE_DEBUG 0
#include "debug.h"

#ifndef SUIT_WORKER_STACKSIZE
/* allocate stack needed to do manifest validation */
#define SUIT_WORKER_STACKSIZE (3 * THREAD_STACKSIZE_LARGE)
#endif

#ifndef SUIT_COAP_WORKER_PRIO
#define SUIT_COAP_WORKER_PRIO THREAD_PRIORITY_MAIN - 1
#endif

/** Maximum size of SUIT manifests processable by the param_update_worker mechanisms */
#ifndef SUIT_MANIFEST_BUFSIZE
#define SUIT_MANIFEST_BUFSIZE   640
#endif

static char _stack[SUIT_WORKER_STACKSIZE];
static char _url[CONFIG_SOCK_URLPATH_MAXLEN];
static ssize_t _size;
static uint8_t _manifest_buf[SUIT_MANIFEST_BUFSIZE];

static mutex_t _worker_lock;
/* PID of the worker thread, guarded by */
static kernel_pid_t _worker_pid = KERNEL_PID_UNDEF;



static int _suit_handle_manifest_buf(const uint8_t *buffer, size_t size)
{
    suit_manifest_t manifest;
    memset(&manifest, 0, sizeof(manifest));

    manifest.urlbuf = _url;
    manifest.urlbuf_len = CONFIG_SOCK_URLPATH_MAXLEN;

    int res;
    if ((res = suit_parse(&manifest, buffer, size)) != SUIT_OK) {
        LOG_INFO("param_update_worker: suit_parse() failed. res=%i\n", res);
        return res;
    }

    return res;
}

static int _suit_handle_url(const char *url)
{
    ssize_t size;
    LOG_INFO("param_update_worker: downloading \"%s\"\n", url);

    if (0) {}
    else if ((strncmp(url, "coap://", 7) == 0) ||
             (IS_USED(MODULE_NANOCOAP_DTLS) && strncmp(url, "coaps://", 8) == 0)) {
        size = nanocoap_get_blockwise_url_to_buf(url,
                                                 CONFIG_SUIT_COAP_BLOCKSIZE,
                                                 _manifest_buf,
                                                 sizeof(_manifest_buf));
    }

    else {
        LOG_WARNING("param_update_worker: unsupported URL scheme!\n)");
        return -ENOTSUP;
    }

    if (size < 0) {
        LOG_INFO("param_update_worker: error getting manifest\n");
        return size;
    }

    LOG_INFO("param_update_worker: got manifest with size %u\n", (unsigned)size);

    return _suit_handle_manifest_buf(_manifest_buf, size);
}

static void *_param_update_worker_thread(void *arg)
{
    (void)arg;

    LOG_INFO("param_update_worker: started.\n");

    int res;
    if (_url[0] == '\0') {
        res = suit_handle_manifest_buf(_manifest_buf, _size);
    } else {
        res = _suit_handle_url(_url);
    }

    if (res == 0) {
        LOG_INFO("param_update_worker: update successful\n");
    }
    else {
        LOG_INFO("param_update_worker: update failed, hdr invalid\n ");
    }

    mutex_unlock(&_worker_lock);
    thread_zombify();
    /* Actually unreachable, given we're in a thread */
    return NULL;
}

/** Reap the zombie worker, or return false if it's still running
 *
 * To call this, the _worker_lock must be held.
 *
 * In the rare case of the worker still running, this releases the
 * _worker_lock, and the caller needs to start over, acquire the _worker_lock
 * and possibly populate the manifest buffer, reap the worker again, and then
 * start the worker thread (with a URL if the manifest was not populated).
 *
 * Otherwise, the worker thread will eventually unlock the mutex.
 * */
static bool _worker_reap(void)
{
    if (_worker_pid != KERNEL_PID_UNDEF) {
        if (thread_kill_zombie(_worker_pid) != 1) {
            /* This will only happen if the SUIT thread runs on a lower
             * priority than the caller */
            LOG_WARNING("Ignoring SUIT trigger: worker is still busy.\n");
            mutex_unlock(&_worker_lock);
            return false;
        }
    }
    return true;
}

void param_update_worker_trigger(const char *url, size_t len)
{
    mutex_lock(&_worker_lock);
    if (!_worker_reap()) {
        return;
    }

    assert(len != 0); /* A zero-length URI is invalid, but _url[0] == '\0' is
                         special to _param_update_worker_thread */
    memcpy(_url, url, len);
    _url[len] = '\0';

    _worker_pid = thread_create(_stack, SUIT_WORKER_STACKSIZE, SUIT_COAP_WORKER_PRIO,
                  THREAD_CREATE_STACKTEST,
                  _param_update_worker_thread, NULL, "suit param update worker");
}

void param_update_worker_trigger_prepared(const uint8_t *buffer, size_t size)
{
    /* As we're being handed the data and the lock (and not just any manifest),
     * we can only accept what was handed out in param_update_worker_try_prepare. */
    (void)buffer;
    assert(buffer = _manifest_buf);

    if (!_worker_reap()) {
        return;
    }

    _url[0] = '\0';
    _size = size;

    if (size == 0) {
        _worker_pid = KERNEL_PID_UNDEF;
        mutex_unlock(&_worker_lock);
        return;
    }

    _worker_pid = thread_create(_stack, SUIT_WORKER_STACKSIZE, SUIT_COAP_WORKER_PRIO,
                  THREAD_CREATE_STACKTEST,
                  _param_update_worker_thread, NULL, "suit param update worker");
}

int param_update_worker_try_prepare(uint8_t **buffer, size_t *size)
{
    if (!mutex_trylock(&_worker_lock)) {
        return -EAGAIN;
    }

    if (*size > SUIT_MANIFEST_BUFSIZE) {
        *size = SUIT_MANIFEST_BUFSIZE;
        mutex_unlock(&_worker_lock);
        return -ENOMEM;
    }

    *buffer = _manifest_buf;
    return 0;
}